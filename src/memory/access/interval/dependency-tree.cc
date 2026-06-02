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
    last_reads(),
    last_write(),
    nwrites(0)
{}

IntervalDependencyTreeNode::IntervalDependencyTreeNode(
    const Hyperrect & h,
    const int k,
    const Color color,
    const Node * inherit
) :
    Base(h, k, color),
    last_reads(),
    last_write(),
    nwrites(0)
{
    this->last_write = inherit->last_write;
    this->last_reads.insert(inherit->last_reads);
}

void
IntervalDependencyTreeNode::update_includes_nwrites(void)
{
    this->nwrites = this->last_write ? 1 : 0;
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
    fprintf(f, "\\nreads=%d\\nwrites=%d", this->last_reads.size(), this->last_write->task ? 1 : 0);
}

void
IntervalDependencyTreeNode::dump_hyperrect_str(FILE * f) const
{
    Base::dump_hyperrect_str(f);

    fprintf(f, "\\\\ reads=%d \\\\ writes=%d", this->last_reads.size(), this->last_write->task ? 1 : 0);
    fprintf(f, "\\\\ nwrites = %d ", this->nwrites);
    fprintf(f, "\\\\ reads = [ ");
    for (const access_t * access : this->last_reads)
        fprintf(f, "%p ", access->task);
    fprintf(f, "]");
}

///////////////////////////////////////
// IntervalDependencyTree            //
///////////////////////////////////////

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
            node->last_reads.clear();
            node->last_write = search.access;
        }
        else if (search.access->mode == ACCESS_MODE_R)
            node->last_reads.push_back(search.access);
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
    return (search.access->mode == ACCESS_MODE_R) && (node->nwrites == 0);
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
            if ((search.access->mode & ACCESS_MODE_W) && node->last_reads.size())
                for (access_t * pred : node->last_reads)
                    __access_precedes(pred, search.access);
            else if (node->last_write)
                __access_precedes(node->last_write, search.access);

            break ;
        }

        default:
        {
            assert(0);
            break ;
        }
    }
}

void
IntervalDependencyTree::link(access_t * access)
{
    Search search;
    search.prepare_resolve(access);
    Base::intersect(search, access->region.interval.segment);
}

void
IntervalDependencyTree::put(access_t * access)
{
    Search search;
    search.prepare_resolve(access);
    Base::insert(search, access->region.interval.segment);

    this->accesses.push_front(access);
}

# undef K

XKRT_NAMESPACE_END
