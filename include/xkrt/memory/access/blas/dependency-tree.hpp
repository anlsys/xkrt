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

#ifndef __DEPENDENCY_TREE_HPP__
# define __DEPENDENCY_TREE_HPP__

# include <xkrt/memory/access/common/lp-tree.hpp>
# include <xkrt/memory/access/dependency-domain.hpp>
# include <xkrt/task/task.hpp>

# include <vector>
# include <unordered_map>

XKRT_NAMESPACE_BEGIN

template<int K>
class KBLASDependencyTreeSearch
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
        KBLASDependencyTreeSearch() {}
        ~KBLASDependencyTreeSearch() {}

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

} /* class KBLASDependencyTreeSearch */;

template <int K>
class KBLASDependencyTreeNode : public LPTree<K, KBLASDependencyTreeSearch<K>>::Node {

    using Base      = typename LPTree<K, KBLASDependencyTreeSearch<K>>::Node;
    using Node      = KBLASDependencyTreeNode<K>;
    using Hyperrect = KHyperrect<K>;
    using Search    = KBLASDependencyTreeSearch<K>;

    public:

        /* last tasks that performed a read access */
        std::vector<access_t *> last_reads;

        /* last task that performed a write access */
        access_t * last_write;

        /* number of writes in all subtrees */
        int nwrites;

    public:

        KBLASDependencyTreeNode<K>(
            const Hyperrect & h,
            const int k,
            const Color color
        ) :
            Base(h, k, color),
            last_reads(),
            last_write(),
            nwrites(0)
        {
        }

        /* a new node from a split, inherit 'src' accesses */
        KBLASDependencyTreeNode<K>(
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

template<int K>
class KBLASDependencyTree : public LPTree<K, KBLASDependencyTreeSearch<K>>, public DependencyDomain
{
    public:
        using Base      = LPTree<K, KBLASDependencyTreeSearch<K>>;
        using Hyperrect = KHyperrect<K>;
        using Node      = KBLASDependencyTreeNode<K>;
        using NodeBase  = typename Base::Node;
        using Search    = KBLASDependencyTreeSearch<K>;

        /* alignment is ld.sizeof_type */
        KBLASDependencyTree(const size_t ld, const size_t sizeof_type) :
            Base(), ld(ld), sizeof_type(sizeof_type) {}
        ~KBLASDependencyTree() {}

        /* alignement for this dep tree */
        const size_t ld;
        const size_t sizeof_type;

    public:

        inline void
        conflicting(
            std::vector<void *> * conflicts,
            access_t * access
        ) {
            // impl assumes this
            assert((access->mode & ACCESS_MODE_R) && !(access->mode & ACCESS_MODE_W));
            assert(access->type == ACCESS_TYPE_SEGMENT || access->type == ACCESS_TYPE_BLAS_MATRIX);

            Search search;
            search.prepare_conflicting(conflicts, access);
            for (const Rect & rect : access->rects())
                Base::intersect(search, rect);
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
            assert(search.access->type == ACCESS_TYPE_SEGMENT || search.access->type == ACCESS_TYPE_BLAS_MATRIX);

            Node * node = reinterpret_cast<Node *>(nodebase);
            assert(node);

            // must check if it intersects, because this node insertion may
            // have been triggered by a splitting a node that do not intersects
            // with the originally inserted rectangle
            for (const Rect & rect : search.access->rects())
            {
                if (rect.intersects(node->hyperrect))
                {
                    if (search.access->mode & ACCESS_MODE_W)
                    {
                        node->last_reads.clear();
                        node->last_write = search.access;
                    }
                    else if (search.access->mode == ACCESS_MODE_R)
                        node->last_reads.push_back(search.access);
                    break ;
                }
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

            Node * node = reinterpret_cast<Node *>(nodebase);
            assert(node);
            assert(node->hyperrect.intersects(h));

            switch (search.type)
            {
                case (Search::Type::SEARCH_TYPE_RESOLVE):
                {
                    if ((search.access->mode & ACCESS_MODE_W) && node->last_reads.size())
                    {
                        for (access_t * pred : node->last_reads)
                            __access_precedes(pred, search.access);
                    }
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

        inline void
        prepare_interval_access_rects(access_t * access, Rect (& rects) [3])
        {
            /* compute the 3 rect for that access in that LP-Tree */
            const INTERVAL_TYPE_T       ptr = (INTERVAL_TYPE_T)      access->host_view.addr;
            const INTERVAL_DIFF_TYPE_T size = (INTERVAL_DIFF_TYPE_T) access->host_view.m;
            interval_to_rects(ptr, size, this->ld, this->sizeof_type, rects);
        }

        inline void
        link(access_t * access, std::span<Rect> rects)
        {
            assert(access->type == ACCESS_TYPE_SEGMENT || access->type == ACCESS_TYPE_BLAS_MATRIX);

            Search search;
            search.prepare_resolve(access);
            for (Rect & rect : rects)
                Base::intersect(search, rect);
        }

        void
        link(access_t * access)
        {
            assert(access->type == ACCESS_TYPE_BLAS_MATRIX);
            std::span<Rect> rects = access->rects();
            this->link(access, rects);
        }

        void
        link_interval(access_t * access)
        {
            assert(access->type == ACCESS_TYPE_SEGMENT);

            Rect rects[3];
            this->prepare_interval_access_rects(access, rects);

            std::span<Rect, 3> rects_span(rects);
            this->link(access, rects_span);
        }

        inline void
        put(access_t * access, std::span<Rect> rects)
        {
            assert(access->type == ACCESS_TYPE_BLAS_MATRIX || access->type == ACCESS_TYPE_SEGMENT);

            Search search;
            search.prepare_resolve(access);
            for (Rect & rect : rects)
                Base::insert(search, rect);
        }

        void
        put(access_t * access)
        {
            assert(access->type == ACCESS_TYPE_BLAS_MATRIX);
            std::span<Rect> rects_span = access->rects();
            this->put(access, rects_span);
        }

        void
        put_interval(access_t * access)
        {
            assert(access->type == ACCESS_TYPE_SEGMENT);

            Rect rects[3];
            this->prepare_interval_access_rects(access, rects);

            std::span<Rect, 3> rects_span(rects);
            this->put(access, rects_span);
        }

};

using BLASDependencyTree = KBLASDependencyTree<2>;

XKRT_NAMESPACE_END

#endif /* __DEPENDENCY_TREE_HPP__ */
