/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
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

// OpenMP 6.0-alike dependencies

#ifndef __DEPENDENCY_MAP_HPP__
# define __DEPENDENCY_MAP_HPP__

# include <xkrt/support.h>
# include <xkrt/memory/access/dependency-domain.hpp>
# include <xkrt/task/task.hpp>

# include <vector>
# include <unordered_map>

XKRT_NAMESPACE_BEGIN

class DependencyMap : public DependencyDomain
{

    class Node {

        public:

            std::vector<access_t *> last_conc_writes;
            std::vector<access_t *> last_seq_reads;
            access_t *  last_seq_write;

        public:

            Node(
            ) :
                last_conc_writes(8),
                last_seq_reads(8),
                last_seq_write()
            {
                last_conc_writes.clear();
                last_seq_reads.clear();
            }

            ~Node() {}

    }; /* Node */

     private:
        std::unordered_map<const void *, Node> map;

     public:
        DependencyMap(const int n = 4096) : map()
        {
            if (n)
                map.reserve(n);
        }

        ~DependencyMap() {}

    public:

        inline void
        insert_empty_write(const void * handle)
        {
            // create the empty task node
            thread_t * thread = thread_t::get_tls();
            assert(thread);

            constexpr int AC = 1;
            constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT;
            constexpr task_format_id_t fmtid = XKRT_TASK_FORMAT_NULL;
            constexpr size_t task_size = task_compute_size(flags, AC);
            constexpr size_t args_size = 0;

            // TODO: this is leaking
            task_t * extra = (task_t *) malloc(task_size + args_size);
            new (extra) task_t(fmtid, flags);

            task_dep_info_t * dep = TASK_DEP_INFO(extra);
            assert(dep);
            new (dep) task_dep_info_t(AC);

            # if XKRT_SUPPORT_DEBUG
            snprintf(extra->label, sizeof(extra->label), "cw-empty-node");
            # endif

            access_t * accesses = TASK_ACCESSES(extra, flags);
            assert(accesses);
            {
                constexpr access_mode_t         mode        = ACCESS_MODE_VW;
                constexpr access_concurrency_t  concurrency = ACCESS_CONCURRENCY_SEQUENTIAL;
                constexpr access_scope_t        scope       = ACCESS_SCOPE_NONUNIFIED;
                new (accesses + 0) access_t(extra, handle, mode, concurrency, scope);
            }

            // TODO : bellow are really ugly implementation hacks
            // maybe refactor the code to go through the traditionnal runtime routines
            this->link(accesses + 0);

            if (dep->wc.fetch_sub(1, std::memory_order_seq_cst) == 1)
            {
                // all predecessors completed already, we can skip that empty node
            }
            else
            {
                assert(thread->current_task);
                extra->parent = thread->current_task;
                ++thread->current_task->cc;

                this->put(accesses + 0);
            }
        }

        // set all accesses of 'list' as predecessors of 'succ'
        // and remove entries in 'list' that already completed
        static inline void
        link_or_pop(
            std::vector<access_t *> & list,
            // std::list<access_t *> & list,
            access_t * succ
        ) {
            # if 1
            for (auto it = list.begin(); it != list.end(); )
            {
                access_t * pred = *it;

                // return true if pred is not already completed
                if (__access_precedes(pred, succ))
                {
                    ++it;
                }
                // pred completed, we can remove it from the list to dampen
                // future accesses search
                else
                {
                    it = list.erase(it);
                }
            }
            # elif 0
            bool clear = true;
            for (access_t * pred : list)
            {
                if (__access_precedes(pred, succ))
                {
                    clear = false;
                }
            }
            if (clear)
                list.clear();
            # else
            for (access_t * pred : list)
                __access_precedes(pred, succ);
            # endif
        }

        //  access type         depend on
        //  SEQ-R               SEQ-W, CNC-W,  COM-W
        //  CNC-W               SEQ-R, SEQ-W,  COM-W,
        //  COM-W               SEQ-R, SEQ-W, (COM-W), CNC-W
        //  SEQ-W               SEQ-R, SEQ-W,  COM-W,  CNC-W

        inline void
        link(access_t * access)
        {
            // retrieve previous accesses on that handle
            auto it = map.find(access->region.point.handle);

            // if none, no dependencies, return
            if (it == map.end())
                return ;

            // else, set dependencies
            Node & node = it->second;
            bool seq_w_edge_transitive = false;

            // the generated access depends on previous SEQ-R
            if (node.last_seq_reads.size() && (access->mode & ACCESS_MODE_W))
            {
                # if 1
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
                    insert_empty_write(access->region.point.handle);
                }
                // SEQ-W
                else
                # endif
                {
                    link_or_pop(node.last_seq_reads, access);
                    seq_w_edge_transitive = true;
                }
            }

            // the generated access depends on previous CNC-W
            if (node.last_conc_writes.size() && access->concurrency != ACCESS_CONCURRENCY_CONCURRENT)
            {
                # if 1
                if (access->mode & ACCESS_MODE_W)
                # endif
                {
                    link_or_pop(node.last_conc_writes, access);
                    seq_w_edge_transitive = true;
                }
                # if 1
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
                    insert_empty_write(access->region.point.handle);
                }
                # endif
            }

            // the generated access depends on previous SEQ-W (they all do)
            if (node.last_seq_write)
            {
                if (seq_w_edge_transitive)
                {
                    // nothing to do: another edge already ensure that
                    // dependency by transitivity
                }
                else
                {
                    if (__access_precedes(node.last_seq_write, access))
                    {
                        // nothing to do, last writer has not completed
                    }
                    else
                    {
                        // last writer completed already, can remove it
                        node.last_seq_write = NULL;
                    }
                }
            }
            else
            {
                // nothing to do: no previous writers
            }
        }

        inline void
        put(access_t * access)
        {
            // TODO : redundancy check, if we allow redundant dependencies - see
            // https://github.com/cea-hpc/mpc/blob/master/src/MPC_OpenMP/src/mpcomp_task.c#L1274

            // ensure a node exists on that address
            auto result = map.insert({access->region.point.handle, Node()});
            if (result.second)
            {
                // node got inserted
            }
            else
            {
                // node already existed
            }

            Node & node = result.first->second;

            if (access->mode & ACCESS_MODE_W)
            {
                if (access->concurrency == ACCESS_CONCURRENCY_CONCURRENT)
                {
                    node.last_conc_writes.push_back(access);
                }
                else
                {
                    assert(access->concurrency == ACCESS_CONCURRENCY_SEQUENTIAL ||
                            access->concurrency == ACCESS_CONCURRENCY_COMMUTATIVE);

                    node.last_seq_reads.clear();
                    node.last_conc_writes.clear();
                    node.last_seq_write = access;
                }
            }
            else if (access->mode & ACCESS_MODE_R)
            {
                node.last_conc_writes.clear();
                node.last_seq_reads.push_back(access);
            }
        }

};

XKRT_NAMESPACE_END

#endif /* __DEPENDENCY_MAP_HPP__ */
