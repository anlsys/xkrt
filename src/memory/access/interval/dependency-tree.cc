/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, romain.pereira@inria.fr + rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

# include <xkrt/memory/access/interval/dependency-tree.hpp>

# define K 1

XKRT_NAMESPACE_BEGIN

///////////////////////////////////////
// IntervalDependencyTreeNode        //
///////////////////////////////////////

IntervalDependencyTreeNode::IntervalDependencyTreeNode(
    const Hyperrect & h,
    const int k,
    const Color color
) :
    Base(h, k, color),
    last_seq_reads(),
    last_seq_write(),
    nwrites(0)
{}

IntervalDependencyTreeNode::IntervalDependencyTreeNode(
    const Hyperrect & h,
    const int k,
    const Color color,
    const Node * inherit
) :
    Base(h, k, color),
    last_seq_reads(),
    last_seq_write(),
    nwrites(0)
{
    this->last_seq_write = inherit->last_seq_write;
    this->last_seq_reads.insert(inherit->last_seq_reads);
    this->last_conc_writes.insert(inherit->last_conc_writes);
}

void
IntervalDependencyTreeNode::update_includes_nwrites(void)
{
    this->nwrites = (this->last_seq_write ? 1 : 0) + this->last_conc_writes.size();
    FOREACH_CHILD_BEGIN(this, child, k, dir)
    {
        this->nwrites += child->nwrites;
    }
    FOREACH_CHILD_END(this, child, k, dir);
}

void
IntervalDependencyTreeNode::update_includes(void)
{
    Base::update_includes();
    this->update_includes_nwrites();
}

void
IntervalDependencyTreeNode::dump_str(FILE * f) const
{
    Base::dump_str(f);
    fprintf(f, "\\nreads=%d\\nseq_write=%d\\nconc_writes=%d",
        this->last_seq_reads.size(),
        this->last_seq_write ? 1 : 0,
        this->last_conc_writes.size());
}

void
IntervalDependencyTreeNode::dump_hyperrect_str(FILE * f) const
{
    Base::dump_hyperrect_str(f);

    fprintf(f, "\\\\ reads=%d \\\\ seq_write=%d \\\\ conc_writes=%d",
        this->last_seq_reads.size(),
        this->last_seq_write ? 1 : 0,
        this->last_conc_writes.size());
    fprintf(f, "\\\\ nwrites = %d ", this->nwrites);
    fprintf(f, "\\\\ reads = [ ");
    for (const access_t * access : this->last_seq_reads)
        fprintf(f, "%p ", access->task);
    fprintf(f, "]");
}

///////////////////////////////////////
// IntervalDependencyTree            //
///////////////////////////////////////

static inline void
insert_empty_write(
    IntervalDependencyTree * tree,
    runtime_t * runtime,
    access_t * access
) {
    // create the empty task node
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    constexpr task_format_id_t fmtid = XKRT_TASK_FORMAT_NULL;
    constexpr task_flag_bitfield_t flags = TASK_FLAG_ACCESSES;
    constexpr void * args = NULL;
    constexpr size_t args_size = 0;
    constexpr int AC = 1;

    task_t * extra = runtime->task_new(fmtid, flags, args, args_size, AC);

    task_acs_info_t * acs = TASK_ACS_INFO(extra);
    assert(acs);
    new (acs) task_acs_info_t(AC);

    # if XKRT_SUPPORT_DEBUG
    snprintf(extra->label, sizeof(extra->label), "cw-empty-node");
    # endif

    access_t * accesses = TASK_ACCESSES(extra);
    assert(accesses);
    {
        constexpr access_mode_t         mode        = ACCESS_MODE_VW;
        constexpr access_concurrency_t  concurrency = ACCESS_CONCURRENCY_SEQUENTIAL;
        constexpr access_scope_t        scope       = ACCESS_SCOPE_NONUNIFIED;
        new (accesses + 0) access_t(
            extra,
            (uintptr_t) access->host_view.addr,
            (uintptr_t) access->host_view.addr + access->host_view.m,
            mode, concurrency, scope
        );
    }

    // TODO : bellow are really ugly implementation hacks
    // maybe refactor the code to go through the traditionnal runtime routines
    tree->link(runtime, accesses + 0);

    if (acs->wc.fetch_sub(1, std::memory_order_seq_cst) == 1)
    {
        // all predecessors completed already, we can skip that empty node
    }
    else
    {
        assert(thread->current_task);
        extra->parent = thread->current_task;
        ++thread->current_task->cc;

        tree->put(accesses + 0);
    }
}

# if XKRT_SUPPORT_STATS
static inline int
__dependency_tree_access_precedes(
    runtime_t * runtime,
    access_t * pred,
    access_t * succ
) {
    int r = __access_precedes(pred, succ);

    switch (r)
    {
        case (XKRT_TASK_DEPENDENCE_ALREADY_SET):
        case (XKRT_TASK_DEPENDENCE_SKIPPED):
        {
            XKRT_STATS_INCR(runtime->stats.edges.skipped, 1);
            break ;
        }

        case (XKRT_TASK_DEPENDENCE_SET):
        {
            XKRT_STATS_INCR(runtime->stats.edges.set, 1);
            break ;
        }

        default:
            break ;
    }
    return r;
}
# else /* XKRT_SUPPORT_STATS */
#  define __dependency_tree_access_precedes(runtime, pred, succ) __access_precedes(pred, succ)
# endif /* XKRT_SUPPORT_STATS */

void
IntervalDependencyTree::on_insert(
    NodeBase * nodebase,
    Search & search
) {
    assert(search.type == Search::Type::SEARCH_TYPE_RESOLVE);

    Node * node = reinterpret_cast<Node *>(nodebase);
    assert(node);

    if (search.access->region.interval.segment.intersects(node->hyperrect))
    {
        if (search.access->mode & ACCESS_MODE_W)
        {
            if (search.access->concurrency == ACCESS_CONCURRENCY_CONCURRENT)
            {
                node->last_conc_writes.push_back(search.access);
            }
            else
            {
                node->last_seq_reads.clear();
                node->last_conc_writes.clear();
                node->last_seq_write = search.access;
            }
        }
        else if (search.access->mode & ACCESS_MODE_R)
            node->last_seq_reads.push_back(search.access);
    }
}

void
IntervalDependencyTree::on_shrink(
    NodeBase * nodebase,
    const Interval & interval,
    int k
) {
    (void) nodebase;
    (void) interval;
    (void) k;
}

IntervalDependencyTree::Node *
IntervalDependencyTree::new_node(
    Search & search,
    const Hyperrect & h,
    const int k,
    const Color color
) const {
    (void) search;
    return new Node(h, k, color);
}

IntervalDependencyTree::Node *
IntervalDependencyTree::new_node(
    Search & search,
    const Hyperrect & h,
    const int k,
    const Color color,
    const NodeBase * inherit
) const {
    (void) search;
    return new Node(h, k, color, reinterpret_cast<const Node *>(inherit));
}

bool
IntervalDependencyTree::intersect_stop_test(
    NodeBase * nodebase,
    Search & search,
    const Hyperrect & h
) const {
    (void) h;

    Node * node = reinterpret_cast<Node *>(nodebase);
    assert(node);

    assert(search.access);

    switch (search.type)
    {
        case (Search::Type::SEARCH_TYPE_RESOLVE):
            return (search.access->mode == ACCESS_MODE_R) && (node->nwrites == 0);

        case (Search::Type::SEARCH_TYPE_NEEDS_EMPTY_WRITE):
            // stop early if we already found a node that needs the empty write
            return search.needs_empty_write;

        default:
            assert(0);
            return false;
    }
}

void
IntervalDependencyTree::on_intersect(
    NodeBase * nodebase,
    Search & search,
    const Hyperrect & h
) const {

    (void) h;

    assert(nodebase);
    Node * node = reinterpret_cast<Node *>(nodebase);

    switch (search.type)
    {
        case (Search::Type::SEARCH_TYPE_RESOLVE):
        {
            runtime_t * runtime = search.runtime;
            access_t * access = search.access;

            //  access type         depend on
            //  SEQ-R               SEQ-W, CNC-W
            //  CNC-W               SEQ-R, SEQ-W
            //  SEQ-W               SEQ-R, SEQ-W, CNC-W

            // the access depends on previous SEQ-R
            if (node->last_seq_reads.size() && (access->mode & ACCESS_MODE_W))
            {
                for (access_t * pred : node->last_seq_reads)
                    __dependency_tree_access_precedes(runtime, pred, access);
            }

            // the access depends on previous CNC-W
            if (node->last_conc_writes.size() && access->concurrency != ACCESS_CONCURRENCY_CONCURRENT)
            {
                for (access_t * pred : node->last_conc_writes)
                    __dependency_tree_access_precedes(runtime, pred, access);
            }

            // the access depends on previous SEQ-W
            if (node->last_seq_write)
            {
                // if we already set edges from SEQ-R or CNC-W, the edge
                // from SEQ-W is already transitively covered when the
                // access is a SEQ-W; but for SEQ-R and CNC-W, we still
                // need the direct edge
                bool seq_w_edge_transitive =
                    (access->mode & ACCESS_MODE_W) &&
                    (access->concurrency != ACCESS_CONCURRENCY_CONCURRENT) &&
                    (node->last_seq_reads.size() || node->last_conc_writes.size());

                if (!seq_w_edge_transitive)
                    __dependency_tree_access_precedes(runtime, node->last_seq_write, access);
            }

            break ;
        }

        case (Search::Type::SEARCH_TYPE_NEEDS_EMPTY_WRITE):
        {
            access_t * access = search.access;

            // CNC-W following SEQ-R: needs empty write
            if ((access->mode & ACCESS_MODE_W) &&
                access->concurrency == ACCESS_CONCURRENCY_CONCURRENT &&
                node->last_seq_reads.size())
            {
                search.needs_empty_write = true;
            }

            // SEQ-R following CNC-W: needs empty write
            if ((access->mode & ACCESS_MODE_R) &&
                node->last_conc_writes.size())
            {
                search.needs_empty_write = true;
            }

            break ;
        }

        default:
        {
            assert(0);
            break ;
        }
    }
}

//  access type         depend on
//  SEQ-R               SEQ-W, CNC-W
//  CNC-W               SEQ-R, SEQ-W
//  COM-W               SEQ-R, SEQ-W, (COM-W), CNC-W
//  SEQ-W               SEQ-R, SEQ-W,  CNC-W

void
IntervalDependencyTree::link(runtime_t * runtime, access_t * access)
{
    // When a concurrent writer follows sequential readers, or
    // when a sequential reader follows concurrent writers,
    // insert an empty sequential write node in-between to
    // reduce the graph complexity from O(n*m) to O(n+m) edges.

    bool needs_empty_write =
        ((access->mode & ACCESS_MODE_W) && access->concurrency == ACCESS_CONCURRENCY_CONCURRENT) ||
        (access->mode == ACCESS_MODE_R);

    if (needs_empty_write)
    {
        // Detection pass: check if any overlapping tree node has a
        // conflicting concurrency state that requires an empty write
        Search search;
        search.prepare_needs_empty_write(access);
        Base::intersect(search, access->region.interval.segment);

        if (search.needs_empty_write)
        {
            /**
             * seq-r :        O O O          conc-w:        O O O
             *                 \|/                           \|/
             * seq-w:           X            seq-w:           X
             *                 / \                           / \
             * conc-w:        O   O          seq-r:         O   O
             */
            insert_empty_write(this, runtime, access);
        }
    }

    // Normal resolve pass: set dependency edges
    Search search;
    search.prepare_resolve(runtime, access);
    Base::intersect(search, access->region.interval.segment);
}

void
IntervalDependencyTree::put(access_t * access)
{
    Search search;
    search.prepare_resolve(nullptr, access);
    Base::insert(search, access->region.interval.segment);

    this->accesses.push_front(access);
}

# undef K

XKRT_NAMESPACE_END
