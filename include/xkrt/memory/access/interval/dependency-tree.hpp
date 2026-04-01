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

# include <xkrt/memory/access/common/lp-tree.hpp>
# include <xkrt/memory/access/dependency-domain.hpp>
# include <xkrt/task/task.hpp>

# include <vector>
# include <unordered_map>

# define K 1

XKRT_NAMESPACE_BEGIN

class IntervalDependencyTreeSearch
{
    public:
        enum Type
        {
            SEARCH_TYPE_RESOLVE,
            SEARCH_TYPE_CONFLICTING
        };

    public:
        Type type;

        // USED IF TYPE == SEARCH_TYPE_RESOLVE or type == SEARCH_TYPE_CONFLICTING
        access_t * access;

        // USED IF TYPE == SEARCH_TYPE_CONFLICTING
        std::vector<void *> * conflicts;

    public:
        IntervalDependencyTreeSearch() {}
        ~IntervalDependencyTreeSearch() {}

    public:

        void
        prepare_resolve(access_t * access)
        {
            this->type = SEARCH_TYPE_RESOLVE;
            this->access = access;
        }

        void
        prepare_conflicting(
            std::vector<void *> * conflicts,
            access_t * access
        ) {
            this->type = SEARCH_TYPE_CONFLICTING;
            this->conflicts = conflicts;
            this->access = access;
        }

} /* class IntervalDependencyTreeSearch */;

class IntervalDependencyTreeNode : public LPTree<K, IntervalDependencyTreeSearch>::Node {

    using Base      = typename LPTree<K, IntervalDependencyTreeSearch>::Node;
    using Node      = IntervalDependencyTreeNode;
    using Hyperrect = KHyperrect<K>;
    using Search    = IntervalDependencyTreeSearch;

    public:

        /* last accesses that read */
        std::vector<access_t *> last_reads;

        /* last access that wrote */
        access_t * last_write;

        /* number of writes in all subtrees */
        int nwrites;

    public:

        IntervalDependencyTreeNode(
            const Hyperrect & h,
            const int k,
            const Color color
        ) :
            Base(h, k, color),
            last_reads(),
            last_write(),
            nwrites(0)
        {}

        /* a new node from a split, inherit 'src' accesses */
        IntervalDependencyTreeNode(
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
            this->last_reads.insert(
                this->last_reads.end(),
                inherit->last_reads.begin(),
                inherit->last_reads.end()
            );
        }

        ////////////
        // UPDATE //
        ////////////
        inline void
        update_includes_nwrites(void)
        {
            this->nwrites = this->last_write ? 1 : 0;
            FOREACH_CHILD_BEGIN(this, child, k, dir)
            {
                this->nwrites += child->nwrites;
            }
            FOREACH_CHILD_END(this, child, k, dir);
        }

        inline void
        update_includes(void)
        {
            Base::update_includes();
            this->update_includes_nwrites();
        }

        void
        dump_str(FILE * f) const
        {
            Base::dump_str(f);
            fprintf(f, "\\nreads=%zu\\nwrites=%d", this->last_reads.size(), this->last_write->task ? 1 : 0);
        }

        void
        dump_hyperrect_str(FILE * f) const
        {
            Base::dump_hyperrect_str(f);

            fprintf(f, "\\\\ reads=%zu \\\\ writes=%d", this->last_reads.size(), this->last_write->task ? 1 : 0);
            fprintf(f, "\\\\ nwrites = %d ", this->nwrites);
            fprintf(f, "\\\\ reads = [ ");
            for (const access_t * access : this->last_reads)
                fprintf(f, "%p ", access->task);
            fprintf(f, "]");
        }
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

        inline void
        conflicting(
            std::vector<void *> * conflicts,
            access_t * access
        ) {
            // impl assumes this
            assert((access->mode & ACCESS_MODE_R) && !(access->mode & ACCESS_MODE_W));

            Search search;
            search.prepare_conflicting(conflicts, access);

            Base::intersect(search, access->region.interval.segment);
        }

        //////////////
        //  INSERT  //
        //////////////

        inline void
        on_insert(
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

        inline void
        on_shrink(
            NodeBase * nodebase,
            const Interval & interval,
            int k
        ) {
            (void) nodebase;
            (void) interval;
            (void) k;
        }

        Node *
        new_node(
            Search & search,
            const Hyperrect & h,
            const int k,
            const Color color
        ) const {
            (void) search;
            return new Node(h, k, color);
        }

        Node *
        new_node(
            Search & search,
            const Hyperrect & h,
            const int k,
            const Color color,
            const NodeBase * inherit
        ) const {
            (void) search;
            return new Node(h, k, color, reinterpret_cast<const Node *>(inherit));
        }

        //////////////////
        //  INTERSECT   //
        //////////////////
        inline bool
        intersect_stop_test(
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

        inline void
        on_intersect(
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

                case (Search::Type::SEARCH_TYPE_CONFLICTING):
                {
                    if (node->last_write)
                    {
                        assert(search.conflicts);
                        search.conflicts->push_back(node);
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

        void
        link(access_t * access)
        {
            Search search;
            search.prepare_resolve(access);
            Base::intersect(search, access->region.interval.segment);
        }

        void
        put(access_t * access)
        {
            Search search;
            search.prepare_resolve(access);
            Base::insert(search, access->region.interval.segment);

            this->accesses.push_front(access);
        }

};

# undef K

XKRT_NAMESPACE_END

#endif /* __INTERVAL_DEPENDENCY_TREE_HPP__ */

