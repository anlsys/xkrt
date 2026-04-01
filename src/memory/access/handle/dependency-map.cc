/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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

// OpenMP 6.0-alike dependencies

# include <xkrt/memory/access/handle/dependency-map.hpp>

XKRT_NAMESPACE_BEGIN

// Empty writes are a really ugly implementation hack, i think it should
// stay disabled, and programmers must explicitly insert empty nodes

static inline void
insert_empty_write(
    DependencyMap * map,
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
        new (accesses + 0) access_t(extra, access->region.point.handle, mode, concurrency, scope);
    }

    // TODO : bellow are really ugly implementation hacks
    // maybe refactor the code to go through the traditionnal runtime routines
    map->link(runtime, accesses + 0);

    if (acs->wc.fetch_sub(1, std::memory_order_seq_cst) == 1)
    {
        // all predecessors completed already, we can skip that empty node
    }
    else
    {
        assert(thread->current_task);
        extra->parent = thread->current_task;
        ++thread->current_task->cc;

        map->put(accesses + 0);
    }
}

// set all accesses of 'list' as predecessors of 'succ'
// and remove entries in 'list' that already completed
static inline void
link_or_pop(small_vector_t<access_t *> & list, access_t * succ)
{
    int i = 0;
    while (i < list.size())
    {
        if (__access_precedes(list[i], succ) == XKRT_TASK_DEPENDENCE_SKIPPED)
            list.swap_erase(i);
        else
            ++i;
    }
}

//  access type         depend on
//  SEQ-R               SEQ-W, CNC-W,  COM-W
//  CNC-W               SEQ-R, SEQ-W,  COM-W,
//  COM-W               SEQ-R, SEQ-W, (COM-W), CNC-W
//  SEQ-W               SEQ-R, SEQ-W,  COM-W,  CNC-W

void
DependencyMap::link(
    runtime_t * runtime,
    access_t * access
) {
    const void * handle = access->region.point.handle;
    Node * node = map.find(handle);
    if (__builtin_expect(node == nullptr, 0))
        return;

    // else, set dependencies
    bool seq_w_edge_transitive = false;

    // the generated access depends on previous SEQ-R
    if (node->last_seq_reads.size() && (access->mode & ACCESS_MODE_W))
    {
        // CNC-W
        if (access->concurrency == ACCESS_CONCURRENCY_CONCURRENT)
        {
            /**
             * seq-r :        O O O
             *                 \|/
             * seq-w:           X       // <- insert that extra node
             *                 / \
             * conc-w:        O   O     // <- inserting this
             */
            insert_empty_write(this, runtime, access);
            node = map.find(handle);   // re-fetch: insert_empty_write may rehash
            assert(node);
        }
        // SEQ-W
        else
        {
            link_or_pop(node->last_seq_reads, access);
            seq_w_edge_transitive = true;
        }
    }

    // the generated access depends on previous CNC-W
    if (node->last_conc_writes.size() && access->concurrency != ACCESS_CONCURRENCY_CONCURRENT)
    {
        if (access->mode & ACCESS_MODE_W)
        {
            link_or_pop(node->last_conc_writes, access);
            seq_w_edge_transitive = true;
        }
        else
        {
            assert(access->mode & ACCESS_MODE_R);

            /**
             * conc-w:        O O O
             *                 \|/
             * seq-w:           X       // <- insert that extra node
             *                 / \
             * seq-r:         O   O     // <- inserting this
             */
            insert_empty_write(this, runtime, access);  // this will clear `node->last_conc_writes`
            node = map.find(handle); // re-fetch: insert_empty_write may rehash
            assert(node);
        }
    }

    // the generated access depends on previous SEQ-W (they all do)
    if (node->last_seq_write)
    {
        if (seq_w_edge_transitive)
        {
            // nothing to do: another edge already ensure that
            // dependency by transitivity
        }
        else
        {
            if (__access_precedes(node->last_seq_write, access) == XKRT_TASK_DEPENDENCE_SKIPPED)
            {
                // If recording, keep it to detect dependencies with future tasks
                if (node->last_seq_write->task && node->last_seq_write->task->flags & TASK_FLAG_RECORD)
                {
                    // nothing to do
                }
                // else, we can remove it now
                else
                {
                    node->last_seq_write = NULL;
                }
            }
            else
            {
                // nothing to do, last writer has not completed
            }
        }
    }
    else
    {
        // nothing to do: no previous writers
    }
}

void
DependencyMap::put(access_t * access)
{
    // TODO : redundancy check, if we allow redundant dependencies - see
    // https://github.com/cea-hpc/mpc/blob/master/src/MPC_OpenMP/src/mpcomp_task.c#L1274

    // ensure a node exists on that address
    auto [node, _] = map.try_emplace(access->region.point.handle);
    if (access->mode & ACCESS_MODE_W)
    {
        if (access->concurrency == ACCESS_CONCURRENCY_CONCURRENT)
        {
            node->last_conc_writes.push_back(access);
        }
        else
        {
            assert(access->concurrency == ACCESS_CONCURRENCY_SEQUENTIAL ||
                    access->concurrency == ACCESS_CONCURRENCY_COMMUTATIVE);

            node->last_seq_reads.clear();
            node->last_conc_writes.clear();
            node->last_seq_write = access;
        }
    }
    else if (access->mode & ACCESS_MODE_R)
    {
        node->last_seq_reads.push_back(access);
    }
}

XKRT_NAMESPACE_END
