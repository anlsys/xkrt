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

#ifndef __INTERVAL_DEPENDENCY_TREE_HPP__
# define __INTERVAL_DEPENDENCY_TREE_HPP__

# include <xkrt/data-structures/small-vector.h>
# include <xkrt/memory/access/common/lp-tree.hpp>
# include <xkrt/memory/access/dependency-domain.hpp>
# include <xkrt/runtime.h>
# include <xkrt/task/task.hpp>

# define K 1

XKRT_NAMESPACE_BEGIN

class IntervalDependencyTreeSearch
{
    public:
        enum Type
        {
            SEARCH_TYPE_RESOLVE,
            SEARCH_TYPE_NEEDS_EMPTY_WRITE
        };

    public:
        Type type;

        // USED IF TYPE == SEARCH_TYPE_RESOLVE or SEARCH_TYPE_NEEDS_EMPTY_WRITE
        access_t * access;

        // runtime pointer for stats tracking
        runtime_t * runtime;

        // USED IF TYPE == SEARCH_TYPE_NEEDS_EMPTY_WRITE (output flag)
        bool needs_empty_write;

    public:
        IntervalDependencyTreeSearch() {}
        ~IntervalDependencyTreeSearch() {}

    public:

        void
        prepare_resolve(runtime_t * runtime, access_t * access)
        {
            this->type = SEARCH_TYPE_RESOLVE;
            this->runtime = runtime;
            this->access = access;
        }

        void
        prepare_needs_empty_write(access_t * access)
        {
            this->type = SEARCH_TYPE_NEEDS_EMPTY_WRITE;
            this->runtime = nullptr;
            this->access = access;
            this->needs_empty_write = false;
        }


} /* class IntervalDependencyTreeSearch */;

class IntervalDependencyTreeNode : public LPTree<K, IntervalDependencyTreeSearch>::Node {

    using Base      = typename LPTree<K, IntervalDependencyTreeSearch>::Node;
    using Node      = IntervalDependencyTreeNode;
    using Hyperrect = KHyperrect<K>;
    using Search    = IntervalDependencyTreeSearch;

    public:

        /* last accesses that read sequentially */
        small_vector_t<access_t *> last_seq_reads;

        /* last accesses that wrote concurrently */
        small_vector_t<access_t *> last_conc_writes;

        /* last access that wrote sequentially */
        access_t * last_seq_write;

        /* number of writes in all subtrees */
        int nwrites;

    public:

        IntervalDependencyTreeNode(
            const Hyperrect & h,
            const int k,
            const Color color
        );

        /* a new node from a split, inherit 'src' accesses */
        IntervalDependencyTreeNode(
            const Hyperrect & h,
            const int k,
            const Color color,
            const Node * inherit
        );

        ////////////
        // UPDATE //
        ////////////

        void update_includes_nwrites(void);
        void update_includes(void);
        void dump_str(FILE * f) const;
        void dump_hyperrect_str(FILE * f) const;
};

class IntervalDependencyTree : public LPTree<K, IntervalDependencyTreeSearch>, public DependencyDomain
{
    public:
        using Base      = LPTree<K, IntervalDependencyTreeSearch>;
        using Hyperrect = KHyperrect<K>;
        using Node      = IntervalDependencyTreeNode;
        using NodeBase  = typename Base::Node;
        using Search    = IntervalDependencyTreeSearch;

    public:

        /* accesses submitted to the interval tree */
        std::list<access_t *> accesses;

    public:

        /* alignment is ld.sizeof_type */
        IntervalDependencyTree() : Base(), accesses() {}
        ~IntervalDependencyTree() {}

    public:

        //////////////
        //  INSERT  //
        //////////////

        void on_insert(NodeBase * nodebase, Search & search);
        void on_shrink(NodeBase * nodebase, const Interval & interval, int k);

        Node * new_node(
            Search & search,
            const Hyperrect & h,
            const int k,
            const Color color
        ) const;

        Node * new_node(
            Search & search,
            const Hyperrect & h,
            const int k,
            const Color color,
            const NodeBase * inherit
        ) const;

        //////////////////
        //  INTERSECT   //
        //////////////////

        bool intersect_stop_test(
            NodeBase * nodebase,
            Search & search,
            const Hyperrect & h
        ) const;

        void on_intersect(
            NodeBase * nodebase,
            Search & search,
            const Hyperrect & h
        ) const;

        void link(runtime_t * runtime, access_t * access);
        void put(access_t * access);

};

# undef K

XKRT_NAMESPACE_END

#endif /* __INTERVAL_DEPENDENCY_TREE_HPP__ */
