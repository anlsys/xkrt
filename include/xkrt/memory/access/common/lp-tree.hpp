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

//  Usage recommendation
//
//  Implement a new structure on top of a lp-tree, lets say `toto-tree.hpp`
//  In `toto-tree.hpp` , before including `lp-tree.hpp`, explicitely define
//  all of these depending on your need:
//      - LP_TREE_REBALANCE
//      - LP_TREE_CUT_ON_INSERT
//      - LP_TREE_MAINTAIN_SIZE
//      - LP_TREE_MAINTAIN_HEIGHT
//
//  See each variable definition in this file for details.
//
//  If you are scare to break the data structure, or suspect a bug, you should
//  set `LP_TREE_ENABLE_COHERENCY_CHECKS`.  It adds several coherency checks
//  on each operations, but severely slowdowns the execution making most operations O(n^2).


// TODO PERFORMANCES IDEA
//  - remove the use of 'virtual' to replace with template 'node' type and inlined functions

#ifndef __LP_TREE_H__
# define __LP_TREE_H__

// tree assert, must be called within a member function
# ifndef LP_TREE_ENABLE_COHERENCY_CHECKS
#  define LP_TREE_ENABLE_COHERENCY_CHECKS 0
# endif
# if LP_TREE_ENABLE_COHERENCY_CHECKS
#  pragma message("Unset `LP_TREE_ENABLE_COHERENCY_CHECKS` for max performance")
#  define tassert(expr)                                                         \
    do {                                                                        \
        if (!(expr))                                                            \
        {                                                                       \
            this->export_pdf("lp-tree");                                       \
            fprintf(stdout, "%s:%d: assertion `" #expr "` failed in `%s`\n",    \
                   __FILE__,__LINE__,__func__);                                 \
            abort();                                                            \
        }                                                                       \
    } while (0)
# else /* LP_TREE_ENABLE_COHERENCY_CHECKS */
#  define tassert(ignore) ((void)0)
# endif /* LP_TREE_ENABLE_COHERENCY_CHECKS */

# include <cassert>
# include <cstdio>
# include <cstring>
# include <cstdlib>
# include <vector>
# include <climits>

# include <functional>
using namespace std::placeholders;

# include <type_traits>

# include <ostream>
# include <iostream>

# include <cstdint>

static inline int
khp_log2(unsigned int n)
{
    return 31 - __builtin_clz(n);
}

static inline int
khp_twopow(int n)
{
    return (1 << n);
}

# include <xkrt/utils/min-max.h>
# include <xkrt/memory/access/common/hyperrect.hpp>
# include <xkrt/sync/direction.h>

# define FOREACH_K_CHILD_BEGIN(N, C, I, D)                              \
do {                                                                    \
    for (int D = LEFT ; D < DIRECTION_MAX ; ++D)                        \
    {                                                                   \
        Node * C = reinterpret_cast<Node *>(N->st[I].children[D]);      \
        if (C)                                                          \
        {
# define FOREACH_K_CHILD_END(N, C, I, D)                                \
        }                                                               \
    }                                                                   \
} while (0)

# define FOREACH_CHILD_BEGIN(N, C, I, D)                                \
do {                                                                    \
    for (int I = 0 ; I < K ; ++I)                                       \
    {                                                                   \
        FOREACH_K_CHILD_BEGIN(N, C, I, D)                               \
        {
# define FOREACH_CHILD_END(N, C, I, D)                                  \
        }                                                               \
        FOREACH_K_CHILD_END(N, C, I, D);                                \
    }                                                                   \
} while (0)


typedef enum    Color
{
    BLACK   = 0,
    RED     = 1
}               Color;

/**
 *  K is the number of dimensions
 *  T is search type
 *  C is whether to cut included nodes or not
 */
template<
    int K,
    typename T,
    bool REBALANCE       = false,
    bool CUT_ON_INSERT   = false,
    bool MAINTAIN_SIZE   = false,
    bool MAINTAIN_HEIGHT = false
>
class LPTree {

    // REBALANCE - Whether to automatically rebalance the tree.  This only
    // makes sense if cutting the tree.  If this is set, you should also set
    // `MAINTAIN_SIZE` and `MAINTAIN_HEIGHT` to avoid O(n) on each imbalance
    // checks

    // CUT_ON_INSERT - to cut subtree nodes when inserting new nodes that
    // passes the `should_cut` test

    // MAINTAIN_SIZE and MAINTAIN_HEIGHT - whether 'height' and 'size' should
    // be computed lazily (i.e. O(n) on each call) or maintained after
    // insertion/rotations

    // `CUT_ON_INSERT` is set but not `REBALANCE`
    // it means the tree might get imbalanced if cutting occurs
    static_assert(!CUT_ON_INSERT || (CUT_ON_INSERT && REBALANCE));


    // `REBALANCE` is set but not `MAINTAIN_SIZE` and `MAINTAIN_HEIGHT`
    // Set them too to avoid O(n) on each operation
    static_assert(!REBALANCE || (REBALANCE && MAINTAIN_SIZE && MAINTAIN_HEIGHT));

    public:

        using Hyperrect = KHyperrect<K>;

        class Node {

            public:

                /* a node subtree */
                typedef union
                {
                    Node * children[2];
                    struct {
                        Node * left;
                        Node * right;
                    };
                } subtree_t;

                /* node's parent */
                Node * parent;

                /* node's color in each subtree */
                Color colors[K];

                /* the rect represented by this node */
                Hyperrect hyperrect;

                /* the dimension represented by this node */
                int k;

                /* node's child */
                subtree_t st[K];

                /* includes metadata */
                template <bool S, bool H>
                struct includes_t {};

                template<>
                struct includes_t<true, true> {

                    // subtree englobing rect
                    Hyperrect hyperrect;

                    // subtree number of elements
                    int size[K];

                    // subtree height
                    int height[K];
                };

                template<>
                struct includes_t<true, false> {
                    Hyperrect hyperrect;
                    int size[K];
                };

                template<>
                struct includes_t<false, true> {
                    Hyperrect hyperrect;
                    int height[K];
                    int total_height;
                };

                template<>
                struct includes_t<false, false> {
                    Hyperrect hyperrect;
                };

                includes_t <MAINTAIN_SIZE, MAINTAIN_HEIGHT> includes;

                #if LP_TREE_ENABLE_COHERENCY_CHECKS
                struct {
                    int id;
                } checks;
                #endif /* LP_TREE_ENABLE_COHERENCY_CHECKS */

            public:

                Node() :
                    parent(nullptr),
                    colors{BLACK}
                {}

                Node(
                    const Hyperrect & h,
                    const int k,
                    const Color color
                ) :
                    parent(nullptr),
                    colors{BLACK},
                    hyperrect(h),
                    k(k)
                {
                    memset(this->st, 0, sizeof(this->st));

                    memcpy(&this->includes.hyperrect, &h, sizeof(Hyperrect));

                    if constexpr (MAINTAIN_SIZE)
                    {
                        memset(this->includes.size, 0, sizeof(this->includes.size));
                        this->includes.size[k] = 1;
                    }

                    if constexpr (MAINTAIN_HEIGHT)
                    {
                        memset(this->includes.height, 0, sizeof(this->includes.height));
                        for (int i = k ; i < K ; ++i)
                            this->includes.height[i] = 1;
                        this->includes.total_height = 1;
                    }

                    this->colors[k] = color;
                }

                virtual ~Node() {}

                ///////////////
                // Utilities //
                ///////////////

                inline Node *
                get_child(int k, Direction dir) const
                {
                    return reinterpret_cast<Node *>(this->st[k].children[dir]);
                }

                // size
                // TODO : maintaining the size (n-k) per k-tree is a bothersome O(K²)
                // It is currently used to detecting imbalance on a k-subtree
                //
                // Another way would to be maintain the size (n) for the entire b-tree,
                // and if load imbalance is detected - h >=2*K*log(n) - then compute
                // the n-k's and rebalance where it needs
                inline void
                update_includes_size(void)
                requires MAINTAIN_SIZE
                {
                    for (int k = 0 ; k < K ; ++k)
                        this->includes.size[k] = 0;
                    this->includes.size[this->k] = 1;

                    for (int k = this->k ; k < K ; ++k)
                    {
                        for (int kk = 0 ; kk < K ; ++kk)
                        {
                            int nl = this->st[kk].left  ? this->st[kk].left->includes.size[k]  : 0;
                            int nr = this->st[kk].right ? this->st[kk].right->includes.size[k] : 0;
                            this->includes.size[k] += nr + nl;
                        }
                    }
                }

                int
                size(const int k) const
                {
                    if constexpr (MAINTAIN_SIZE)
                    {
                        return this->includes.size[k];
                    }
                    else
                    {
                        int s = (k == this->k) ? 1 : 0;
                        for (int kk = 0 ; kk < K ; ++kk)
                        {
                            int nl = this->st[kk].left  ? this->st[kk].left->size(k)  : 0;
                            int nr = this->st[kk].right ? this->st[kk].right->size(k) : 0;
                            s += nr + nl;
                        }
                        return s;
                    }
                }

                inline int
                size(void) const
                {
                    int s = 0;
                    for (int k = 0 ; k < K ; ++k)
                        s += this->size(k);
                    return s;
                }

                // height
                inline void
                update_includes_height(void)
                requires MAINTAIN_HEIGHT
                {
                    for (int k = 0 ; k < K ; ++k)
                    {
                        // get left height
                        int hleft, thleft;
                        if (this->st[k].left)
                        {
                            hleft  = this->st[k].left->includes.height[k];
                            thleft = this->st[k].left->includes.total_height;
                        }
                        else
                        {
                            hleft  = 0;
                            thleft = 0;
                        }

                        // get right height
                        int hright, thright;
                        if (this->st[k].right)
                        {
                            hright  = this->st[k].right->includes.height[k];
                            thright = this->st[k].right->includes.total_height;
                        }
                        else
                        {
                            hright  = 0;
                            thright = 0;
                        }

                        // get max height
                        this->includes.height[k]    = 1 + MAX(hleft, hright);
                        this->includes.total_height = 1 + MAX(thleft, thright);
                    }
                }

                int
                height(int k) const
                {
                    if constexpr(MAINTAIN_HEIGHT)
                    {
                        return this->includes.height[k];
                    }
                    else
                    {
                        int hleft  = this->st[k].left  ? this->st[k].left->height(k)  : 0;
                        int hright = this->st[k].right ? this->st[k].right->height(k) : 0;
                        return 1 + MAX(hleft, hright);
                    }
                }

                inline int
                height(void) const
                {
                    if constexpr(MAINTAIN_HEIGHT)
                    {
                        return this->includes.total_height;
                    }
                    else
                    {
                        int thleft  = this->st[k].left  ? this->st[k].left->height()  : 0;
                        int thright = this->st[k].right ? this->st[k].right->height() : 0;
                        return 1 + MAX(thleft, thright);
                    }
                }

                inline void
                update_includes_interval(void)
                {
                    for (int k = 0 ; k < K ; ++k)
                    {
                        this->includes.hyperrect[k].a = this->hyperrect[k].a;
                        this->includes.hyperrect[k].b = this->hyperrect[k].b;

                        FOREACH_CHILD_BEGIN(this, child, kk, dir)
                        {
                            this->includes.hyperrect[k].a = MIN(
                                 this->includes.hyperrect[k].a,
                                child->includes.hyperrect[k].a
                            );

                            this->includes.hyperrect[k].b = MAX(
                                 this->includes.hyperrect[k].b,
                                child->includes.hyperrect[k].b
                            );
                        }
                        FOREACH_CHILD_END(this, child, kk, dir);
                    }
                }

                virtual inline void
                update_includes(void)
                {
                    this->update_includes_interval();

                    if constexpr (MAINTAIN_SIZE)
                        this->update_includes_size();

                    if constexpr (MAINTAIN_HEIGHT)
                        this->update_includes_height();
                }

                void
                dump(FILE * f) const
                {
                    // dump the node
                    const char * COLORS[] = {
                        "#000000",
                        "#EE3333",
                        "#3333EE",
                        "#33EE33",
                        "#FFFF00",
                        "#FF00FF",
                        "#00FFFF",
                    };
                    const char * color = COLORS[this->colors[this->k] == BLACK ? 0 : this->k+1];

                    // fprintf(f, "    N%p[fontcolor=\"#ffffff\", label=\"--- node ---\\n", this);
                    // this->dump_str(f);
                    // fprintf(f, "\", style=filled, fillcolor=\"%s\"] ;\n", color);

                    fprintf(f, "    N%p[fontcolor=\"#ffffff\", label=\"", this);
                    this->dump_str(f);
                        fprintf(f, "\", shape=square, style=filled, fillcolor=\"%s\"] ;\n", color);

                    // dump each child
                    for (int k = 0 ; k < K ; ++k)
                    {
                        for (int d = 0 ; d < DIRECTION_MAX ; ++d)
                        {
                            Node * child = reinterpret_cast<Node *>(this->st[k].children[d]);
                            if (child)
                            {
                                child->dump(f);
                                fprintf(f, "    N%p->N%p ; \n", this, child);
                            }
                            # if 0
                            else
                            {
                                int idx = 2 * k + d;
                                void * nullnode = ((char *) this) + idx + 1;
                                fprintf(f, "    N%p[fontcolor=\"#ffffff\", label=\".\", shape=square, style=filled, fillcolor=\"%s\"] ;\n", nullnode, COLORS[0]);
                                fprintf(f, "    N%p->N%p ; \n", this, nullnode);
                            }
                            # endif
                        }
                    }
                }

                virtual void
                dump_str(FILE * f) const
                {
                    char rect[1024];
                    this->hyperrect.tostring(rect, sizeof(rect));

                    char include_rect[1024];
                    this->includes.hyperrect.tostring(include_rect, sizeof(include_rect));

                    if (K == 2)
                    {
                        fprintf(f, "k=%d\\n%s\\n\\n--- includes ---\\n%s\\nsize=%d\\nheight=%d",
                            this->k,
                            rect,
                            include_rect,
                            this->size(),
                            this->height()
                        );
                    }
                }

                void
                dump_hyperrect(FILE * f) const
                {
                    assert(K == 1 || K == 2);
                    if (K == 1)
                    {
                        fprintf(f, "    \\draw (" INTERVAL_TYPE_MODIFIER ",-" INTERVAL_TYPE_MODIFIER ") rectangle (" INTERVAL_TYPE_MODIFIER ",-" INTERVAL_TYPE_MODIFIER ") node[midway] {[" INTERVAL_TYPE_MODIFIER ".." INTERVAL_TYPE_MODIFIER "[};\n",
                                this->hyperrect[0].a, (INTERVAL_TYPE_T) 0,
                                this->hyperrect[0].b, (INTERVAL_TYPE_T) 2,
                                this->hyperrect[0].a, this->hyperrect[0].b
                        );
                    }
                    else if (K == 2)
                    {
                        fprintf(f, "    \\draw [line width=2mm] (" INTERVAL_TYPE_MODIFIER ",-" INTERVAL_TYPE_MODIFIER ") rectangle (" INTERVAL_TYPE_MODIFIER ",-" INTERVAL_TYPE_MODIFIER ") node[midway,scale=10] {",
                            this->hyperrect[1].a, this->hyperrect[0].a,
                            this->hyperrect[1].b, this->hyperrect[0].b
                        );
                        this->dump_hyperrect_str(f);
                        fprintf(f, "};\n");
                    }

                    FOREACH_CHILD_BEGIN(this, child, k, dir)
                    {
                        child->dump_hyperrect(f);
                    }
                    FOREACH_CHILD_END(this, child, k, dir);
                }

                virtual void
                dump_hyperrect_str(FILE * f) const
                {
                    fprintf(f, "[" INTERVAL_TYPE_MODIFIER ".." INTERVAL_TYPE_MODIFIER "[ x [" INTERVAL_TYPE_MODIFIER ".." INTERVAL_TYPE_MODIFIER "[",
                        this->hyperrect[0].a, this->hyperrect[0].b,
                        this->hyperrect[1].a, this->hyperrect[1].b
                    );
                }

        }; /* class Node */

    /* class tree */
    public:

        /* Root node */
        Node * root;

    private:

        /* List of cut-out branches whose subtree requires deletion from memory */
        std::vector<Node *> limbs;

    public:
        LPTree() :
            root(nullptr),
            limbs()
        {}

        inline void
        subtree_delete(Node * node)
        {
            if (node == nullptr)
                return ;

            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                subtree_delete(child);
            }
            FOREACH_CHILD_END(node, child, k, dir);

            delete node;
        }

        inline void
        cut(Node * node)
        {
            this->limbs.push_back(node);
        }

        inline void
        cut(Node * parent, int k, int dir)
        {
            this->cut(parent->st[k].children[dir]);
            parent->st[k].children[dir] = nullptr;
        }

        inline void
        garbage_collector_run(void)
        {
            for (Node * & node : this->limbs)
                subtree_delete(node);
            this->limbs.clear();
        }

        virtual ~LPTree()
        {
            subtree_delete(this->root);
            this->garbage_collector_run();
        }

        ///////////
        // UTILS //
        ///////////
        template <typename N>
        void
        foreach_k_child(
            N * root,
            int k,
            std::function<void(N *)> f
        ) {
            static_assert(std::is_base_of<Node, N>::value);

            for (int i = 0 ; i < 2 ; ++i)
            {
                N * child = reinterpret_cast<N *>(root->st[k].children[i]);
                if (child)
                {
                    f(child);
                    foreach_k_child(child, k, f);
                }
            }
        }

        void
        foreach_node(
            Node * root,
            std::function<void(Node *, void *)> f,
            void * args
        ) const {
            f(root, args);
            FOREACH_CHILD_BEGIN(root, child, k, dir)
            {
                foreach_node(child, f, args);
            }
            FOREACH_CHILD_END(root, child, k, dir);
        }

        void
        foreach_node(
            std::function<void(Node *, void *)> f,
            void * args
        ) {
            this->foreach_node(this->root, f, args);
        }

        void
        foreach_node_until(
            Node * root,
            std::function<void(Node *, void *, bool &)> f,
            void * args,
            bool & stop
        ) const {
            f(root, args, stop);
            if (stop) return ;
            FOREACH_CHILD_BEGIN(root, child, k, dir)
            {
                foreach_node_until(child, f, args, stop);
                if (stop) return ;
            }
            FOREACH_CHILD_END(root, child, k, dir);
        }

        void
        foreach_node_until(
            std::function<void(Node *, void *, bool &)> f,
            void * args
        ) {
            bool stop = false;
            this->foreach_node_until(this->root, f, args, stop);
        }

        int
        height(void) const
        {
            return (this->root ? this->root->height() : 0);
        }

        int
        size(void) const
        {
            return this->root ? this->root->size() : 0;
        }

        void
        export_pdf(const char * label) const
        {
            char filename[512];

            snprintf(filename, sizeof(filename), "%s-tree.dot", label);
            FILE * file = fopen(filename, "w");
            this->dump(file);
            fclose(file);

            snprintf(filename, sizeof(filename),
                    "dot -Tpdf -Gdpi=600 %s-tree.dot > %s-tree.pdf",
                    label, label
            );
            int r = system(filename);
            if (r)
                fprintf(stderr, "dot failed\n");

            snprintf(filename, sizeof(filename), "%s-rect.tex", label);
            file = fopen(filename, "w");
            this->dump_hyperrect(file);
            fclose(file);

            snprintf(filename, sizeof(filename),
                    "pdflatex -interaction=nonstopmode %s-rect.tex > /dev/null 2>&1",
                    label
            );
            r = system(filename);
            if (r)
                fprintf(stderr, "failed: `%s`\n", filename);
        }

        //////////////////
        //  INTERSECT   //
        //////////////////
        inline void
        intersect_from(
            T & t,
            const Hyperrect & h,
            Node * node
        ) const {

            if (node == nullptr || !h.intersects(node->includes.hyperrect))
                return ;

            if (this->intersect_stop_test(node, t, h))
                return ;

            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                this->intersect_from(t, h, child);
            }
            FOREACH_CHILD_END(node, child, k, dir);

            // in-order traversal
            if (h.intersects(node->hyperrect))
                this->on_intersect(node, t, h);
        }

        inline void
        intersect(
            T & t,
            const Hyperrect & h
        ) const {
            if (h.is_empty())
                return ;
            this->intersect_from(t, h, this->root);
        }

        //////////////
        //  INSERT  //
        //////////////

        // redundant updates
        void
        update(Node * node)
        {
            while (1)
            {
                node->update_includes();
                if (node->parent)
                    node = node->parent;
                else
                    break ;
            }
        }

        /**
         *      C              A
         *     / \            / \
         *    A   E    <-    B   C
         *   / \                / \
         *  B   D              D   E
         */
        void
        rotate_left(Node * A, int k)
        {
            tassert(A->st[k].right);

         // Node * B = A->st[k].left;
            Node * C = A->st[k].right;
            Node * D = C->st[k].left;
         // Node * E = C->st[k].right;

//            printf("rl(%d, %d, %d, %d, %d)\n", A->parent ? A->parent->k : -1, A->k, C->k, D ? D->k : -1, k);

            C->st[k].left  = A;
         // C->right = E;
         // A->left  = B;
            A->st[k].right = D;

            C->parent = A->parent;
            if (A->parent == nullptr)
                this->root = C;
            else if (A->parent->st[k].left == A)
                A->parent->st[k].left = C;
            else
                A->parent->st[k].right = C;

         // B->parent = A;
            A->parent = C;
            if (D)
                D->parent = A;
         // E->parent = C;

         // B->update_includes();
         // D->update_includes();
         // E->update_includes();
            A->update_includes();
            C->update_includes();
        }


        /**
         *      A              B
         *     / \            / \
         *    B   C    ->    D   A
         *   / \                / \
         *  D   E              E   C
         */
        void
        rotate_right(Node * A, int k)
        {
            Node * B = A->st[k].left;
         // Node * C = A->st[k].right;
         // Node * D = B->st[k].left;
            Node * E = B->st[k].right;

//            printf("rr(%d, %d, %d, %d, %d)\n", A->parent ? A->parent->k : -1, A->k, B->k, E ? E->k : -1, k);

            // UPDATE LINKS

         // B->st[k].left  = D;
            B->st[k].right = A;
            A->st[k].left  = E;
         // A->st[k].right = C;

            B->parent = A->parent;
            if (A->parent == nullptr)
                this->root = B;
            else if (A->parent->st[k].left == A)
                A->parent->st[k].left = B;
            else
                A->parent->st[k].right = B;

            if (E)
                E->parent = A;
         // C->parent = A;
            A->parent = B;
         // D->parent = B;

         // E->update_includes();
         // C->update_includes();
         // D->update_includes();
            A->update_includes();
            B->update_includes();
        }

        inline void
        balance_fixup(int k, Node * z)
        {
            tassert(z->colors[k] == RED);

            // Traditional red-black tree balancing...
            while (z->parent && z->parent->colors[k] == RED)
            {
                // .. but stopping on the k-root
                if (z->parent->parent && z->parent->parent->k < k)
                {
                    tassert(z->colors[k]            == RED);
                    tassert(z->parent->colors[k]    == RED);
                    z->parent->colors[k] = BLACK;
                    break ;
                }

                if (z->parent == z->parent->parent->st[k].left)
                {
                    Node * y = z->parent->parent->st[k].right;
                    if (y && y->colors[k] == RED)
                    {
                        z->parent->colors[k] = BLACK;
                        y->colors[k] = BLACK;
                        z->parent->parent->colors[k] = RED;
                        z = z->parent->parent;
                    }
                    else
                    {
                        if (z == z->parent->st[k].right)
                        {
                            z = z->parent;
                            this->rotate_left(z, k);
                        }
                        z->parent->colors[k] = BLACK;
                        z->parent->parent->colors[k] = RED;
                        this->rotate_right(z->parent->parent, k);
                    }
                }
                else
                {
                    Node * y = z->parent->parent->st[k].left;

                    if (y && y->colors[k] == RED)
                    {
                        z->parent->colors[k] = BLACK;
                        y->colors[k] = BLACK;
                        z->parent->parent->colors[k] = RED;
                        z = z->parent->parent;
                    }
                    else
                    {
                        if (z == z->parent->st[k].left)
                        {
                            z = z->parent;
                            this->rotate_right(z, k);
                        }
                        z->parent->colors[k] = BLACK;
                        z->parent->parent->colors[k] = RED;
                        this->rotate_left(z->parent->parent, k);
                    }
                }
            }
            this->root->colors[k] = BLACK;
        }

        inline void
        insert_finalize(
            Node * node,
            T & t
        ) {
            this->on_insert(node, t);
            this->update(node);
        }

        inline void
        insert_fixup(
            T & t,
            const Hyperrect & h,
            Node * parent,
            int k,
            Direction dir,
            const Node * inherit
        ) {
            Node * node;
            if (inherit)
                node = this->new_node(t, h, k, RED, inherit);
            else
                node = this->new_node(t, h, k, RED);
            tassert(node);

            parent->st[k].children[dir] = node;
            node->parent = parent;

            // inserting a new k-subtree, this k-root is black
            if (parent->k < k)
                node->colors[k] = BLACK;
            // rebalance the k-subtree
            else
                this->balance_fixup(k, node);

            this->insert_finalize(node, t);
        }

        static inline void
        compress(Node * root, int k, int m)
        {
            Node * tmp = root->st[k].right;

            for (int i = 0; i < m ; ++i)
            {
                Node * oldtmp = tmp;
                tmp = tmp->st[k].right;
                root->st[k].right = tmp;
                oldtmp->st[k].right = tmp->st[k].left;
                tmp->st[k].left = oldtmp;
                root = tmp;
                tmp = tmp->st[k].right;
            }
        }

        static inline int
        vine_to_rbtree(Node * root, int k, int n)
        {
            int h = khp_log2(n + 1);
            int m = khp_twopow(h) - 1;

            compress(root, k, n - m);

            for (m = m / 2; m > 0; m /= 2)
                compress(root, k, m);

            return h;
        }

        static void
        rbtree_to_vine(Node * root, int k)
        {
            Node * tmp = root->st[k].right;

            while (tmp)
            {
                if (tmp->st[k].left)
                {
                    Node * oldtmp = tmp;
                    tmp = tmp->st[k].left;
                    oldtmp->st[k].left = tmp->st[k].right;
                    tmp->st[k].right = oldtmp;
                    root->st[k].right = tmp;
                }
                else
                {
                    root = tmp;
                    tmp = tmp->st[k].right;
                }
            }
        }

        // fixup the tree that just got rebalanced
        static inline void
        rebalance_fixup(Node * parent, Node * node, int k, int depth, int height)
        {
            if (node == nullptr)
                return ;

            rebalance_fixup(node, node->st[k].left,  k, depth + 1, height);
            rebalance_fixup(node, node->st[k].right, k, depth + 1, height);

            node->parent = parent;
            node->colors[k] = (height == depth) ? RED : BLACK;
            node->update_includes();
        }

        // rebalance the k-subtree using a Day-Stout-Warren algorithm
        inline void
        rebalance(Node * root, int k)
        {
            printf("Rebalancing for k=%d\n", k);
            tassert(k == 0 && K == 0 && "Not implemented when K>1");

            Node pseudo_root;
            pseudo_root.st[k].right = root;

            rbtree_to_vine(&pseudo_root, k);
            int height = vine_to_rbtree(&pseudo_root, k, root->size(k));

            // fixup the tree
            Node * new_root = pseudo_root.st[k].right;
            if (root->parent == nullptr)
                this->root = new_root;
            else
            {
                // k > 0 here
                tassert(0 && "Not implemented when K>1");
            }

            rebalance_fixup(nullptr, new_root, k, 0, height);

# if LP_TREE_ENABLE_COHERENCY_CHECKS
            this->coherency(root->includes.hyperrect);
# endif /* LP_TREE_ENABLE_COHERENCY_CHECKS */
        }

        inline void
        rebalance(Node * root)
        {
            for (int k = 0 ; k < K ; ++k)
            {
                if (this->requires_rebalance(root, k))
                    rebalance(root, k);
            }
        }

        inline void
        rebalance(void)
        {
            this->garbage_collector_run();
            this->rebalance(this->root);
        }

        // heuristic to determine whether the tree needs rebalancing
        template<int KDIM>
        inline int
        requires_rebalance(const int size, const int height) const
        {
            int ideal_height = khp_log2(size + 1);
            // return (height >= 8 && height > 2 * KDIM * ideal_height);
            return (height > 2 * KDIM * ideal_height);
        }

        // if the k-subtree starting at 'root' requires rebalance
        inline int
        requires_rebalance(Node * root, int k)
        {
            const int   size = root->size(k);
            const int height = root->height(k);
            return this->requires_rebalance<1>(size, height);
        }

        // if the btree at 'root' requires rebalance
        inline int
        requires_rebalance(Node * root) const
        {
            const int size = root->size();
            const int height = root->height();
            return this->requires_rebalance<K>(size, height);
        }

        // if 'this' btree requires rebalance
        inline bool
        requires_rebalance(void) const
        {
            return this->root && requires_rebalance(this->root);
        }

        void
        post_insert(const Hyperrect & h)
        {
            if constexpr(REBALANCE)
                if (this->requires_rebalance())
                    this->rebalance();

# if LP_TREE_ENABLE_COHERENCY_CHECKS
            this->coherency(h);
# else
            (void) h;
# endif /* LP_TREE_ENABLE_COHERENCY_CHECKS */
        }

        inline void
        insert_from_cut(
            T & t,
            const Hyperrect & h,
            Node * parent
        ) {
            FOREACH_CHILD_BEGIN(parent, child, k, dir)
            {
                this->cut(parent, k, dir);
            }
            FOREACH_CHILD_END(parent, child, k, dir);

            memcpy(&parent->hyperrect, &h, sizeof(Hyperrect));
            this->insert_finalize(parent, t);
        }

        inline void
        insert_from(
            T & t,
            Hyperrect & h,
            Node * parent,
            int k,
            const Node * inherit
        ) {

            while (k < K)
            {
                # if 0
                // quick-way out, if the rect includes all subrect with an
                // 'out' access, we can discard all children
                if constexpr (CUT_ON_INSERT)
                {
                    if (h.includes(parent->includes.hyperrect, k))
                    {
                        // the includes test is accelerated as we know we are
                        // already matching dimensions <k
                        if (this->should_cut(t, h, parent, k))
                        {
                            // TODO : what if 'node' is not null ?  probably want to
                            // return something to callee for the case (3)
                            this->insert_from_cut(t, h, parent);
                            break ;
                        }
                    }
                }
                # endif

                // case (1)    J << I
                if (h[k].b <= parent->hyperrect[k].a)
                {
                    if (parent->st[k].left == nullptr)
                    {
                        this->insert_fixup(t, h, parent, k, LEFT, inherit);
                        break ;
                    }
                    else
                        parent = parent->get_child(k, LEFT);
                }
                // case (2)     J >> I
                else if (h[k].a >= parent->hyperrect[k].b)
                {
                    if (parent->st[k].right == nullptr)
                    {
                        this->insert_fixup(t, h, parent, k, RIGHT, inherit);
                        break ;
                    }
                    else
                        parent = parent->get_child(k, RIGHT);
                }
                // case (3)     J c I   (or I == J)
                else if (parent->hyperrect[k].a <= h[k].a && h[k].b <= parent->hyperrect[k].b)
                {
                    // I == J
                    if (h[k].a == parent->hyperrect[k].a && h[k].b == parent->hyperrect[k].b)
                    {
                        if (++k == K)
                        {
                            this->insert_finalize(parent, t);
                            break ;
                        }
                    }
                    // J c I
                    else
                    {
                        // assert(K == 1 || K == 2);

                        class ReinsertHyperrect {
                            public:
                                Hyperrect h;
                                Node * inherit;
                            public:
                                ReinsertHyperrect(const Hyperrect & r, const Interval & interval, int k, Node * i) :
                                    h(r),
                                    inherit(i)
                                {
                                    h[k] = interval;
                                    inherit = h.intersects(i->hyperrect) ? i : nullptr;
                                }

                                ~ReinsertHyperrect() {}
                        };

                        std::vector<ReinsertHyperrect> to_reinsert;
                        const Interval intervals[] = {
                            Interval(parent->hyperrect[k].a,                 h[k].a),
                            Interval(                h[k].a,                 h[k].b),
                            Interval(                h[k].b, parent->hyperrect[k].b)
                        };

                        std::function<void(Node *)> f = [this, &f, &intervals, &k, &to_reinsert](Node * node)
                        {
                            assert(node->hyperrect[k].includes(intervals[1]));

                            // reinsert sides
                            if (!intervals[0].is_empty())
                                to_reinsert.push_back(ReinsertHyperrect(node->hyperrect, intervals[0], k, node));
                            if (!intervals[2].is_empty())
                                to_reinsert.push_back(ReinsertHyperrect(node->hyperrect, intervals[2], k, node));

                            // shrink node
                            this->on_shrink(node, intervals[1], k);
                            node->hyperrect[k] = intervals[1];

                            // shrink all child
                            for (int kk = k+1; kk < K ; ++kk)
                            {
                                FOREACH_K_CHILD_BEGIN(node, child, kk, dir)
                                {
                                    f(child);
                                }
                                FOREACH_K_CHILD_END(node, child, kk, dir);
                            }
                        };

                        f(parent);

                        assert(inherit == nullptr);
                        for (ReinsertHyperrect & rr : to_reinsert)
                            this->insert_from(t, rr.h, this->root, 0, rr.inherit);

                        return this->insert_from(t, h, this->root, 0, nullptr);

                    } /* I == J ||  J c I */
                }
                // case (4)     I n J != o is (1) + (2) + (3)
                else
                {
                    //  [           I           ]
                    //          [           J             ]
                    //
                    //          or
                    //
                    //  [           J           ]
                    //          [           I             ]
                    const INTERVAL_TYPE_T a = h[k].a;
                    const INTERVAL_TYPE_T b = h[k].b;

                    // (1)
                    if (h[k].a < parent->hyperrect[k].a)
                    {
                        h[k].b = parent->hyperrect[k].a;
                        this->insert_from(t, h, this->root, 0, inherit);
                        h[k].b = b;
                    }

                    // (2)
                    if (parent->hyperrect[k].b < h[k].b)
                    {
                        h[k].a = parent->hyperrect[k].b;
                        this->insert_from(t, h, this->root, 0, inherit);
                        h[k].a = a;
                    }

                    // (3)
                    {
                        h[k].a = MAX(a, parent->hyperrect[k].a);
                        h[k].b = MIN(b, parent->hyperrect[k].b);
                        this->insert_from(t, h, this->root, 0, inherit);
                        h[k].a = a;
                        h[k].b = b;
                    }

                    break ;
                }
            }
        }

        inline void
        insert(
            T & t,
            Hyperrect & h
        ) {
            if (h.is_empty())
                return ;

            if (this->root == nullptr)
            {
                this->root = this->new_node(t, h, 0, BLACK);
                this->insert_finalize(this->root, t);
            }
            else
            {
                this->insert_from(t, h, this->root, 0, nullptr);
            }

            this->post_insert(h);
        }

        inline void
        clear(void)
        {
            this->cut(this->root);
            this->garbage_collector_run();
            this->root = nullptr;
        }

        // Dump the tree to the given file
        void
        dump(FILE * f) const
        {
            fprintf(f, "digraph g {\n");
            if (this->root)
                this->root->dump(f);
            fprintf(f, "}\n");
        }

        // Dump represented hyperrect to the given file
        void
        dump_hyperrect(FILE * f) const
        {
            fprintf(f, "\\documentclass[crop,tikz]{standalone}\n");
            fprintf(f, "\\usetikzlibrary{shapes.multipart}\n");
            fprintf(f, "\\begin{document}\n");
            fprintf(f, "\\begin{tikzpicture}[every text node part/.style={align=center}]\n"  );

            if constexpr (K == 1 || K == 2)
            {
                if (this->root)
                    this->root->dump_hyperrect(f);
            }
            else
            {
                fprintf(f, " Output for K=%d is not supported", K);
            }

            fprintf(f, "  \\end{tikzpicture}\n");
            fprintf(f, "\\end{document}\n");
        }

#if LP_TREE_ENABLE_COHERENCY_CHECKS

    public:
        //////////////////////
        // Coherency checks //
        //////////////////////
        void
        coherency_k(Node * node) const
        {
            // if k < k', then k'-nodes must be children of k-nodes
            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                tassert(node->k <= child->k);
                tassert(child->k == k);
                coherency_k(child);
            }
            FOREACH_CHILD_END(node, child, k, dir);
        }

        void
        coherency_single_path_reset(Node * node) const
        {
            node->checks.id = 0;
            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                coherency_single_path_reset(child);
            }
            FOREACH_CHILD_END(node, child, k, dir);
        }

        void
        coherency_single_path_set(Node * node) const
        {
            tassert(node->checks.id == 0);
            node->checks.id = 1;
            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                coherency_single_path_set(child);
            }
            FOREACH_CHILD_END(node, child, k, dir);
        }

        void
        coherency_single_path(Node * node) const
        {
            // graph must be a tree
            coherency_single_path_reset(node);
            coherency_single_path_set(node);
        }

        void
        coherency_hyperrect_includes_check(Node * ref, void * args) const
        {
            Node * root = (Node *) args;
            tassert(root->includes.hyperrect.includes(ref->hyperrect));
        }

        void
        coherency_hyperrect_includes_foreach(Node * node, void * args) const
        {
            (void) args;
            auto f = std::bind(&LPTree<K, T>::coherency_hyperrect_includes_check, this, _1, _2);
            foreach_node(node, f, node);
        }

        void
        coherency_hyperrect_includes(Node * root) const
        {
            auto f = std::bind(&LPTree<K, T>::coherency_hyperrect_includes_foreach, this, _1, _2);
            foreach_node(root, f, root);
        }

        void
        coherency_hyperrect_disjoint_compare(Node * ref, void * args) const
        {
            Node * node = (Node *) args;
            tassert(node == ref || !node->hyperrect.intersects(ref->hyperrect));
        }

        void
        coherency_hyperrect_disjoint_for(Node * node, void * args) const
        {
            Node * root = (Node *) args;
            auto f = std::bind(&LPTree<K, T>::coherency_hyperrect_disjoint_compare, this, _1, _2);
            foreach_node(root, f, node);
        }

        void
        coherency_hyperrect_disjoint(Node * root) const
        {
            auto f = std::bind(&LPTree<K, T>::coherency_hyperrect_disjoint_for, this, _1, _2);
            foreach_node(root, f, root);
        }

        void
        coherency_color(Node * node) const
        {
            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                // a node is black or red
                tassert(node->colors[k] == BLACK || node->colors[k] == RED);

                // children of a red nodes are black
                if (node->colors[k] == RED)
                {
                    tassert(child->colors[k] == BLACK);
                }

                coherency_color(child);
            }
            FOREACH_CHILD_END(node, child, k, dir);
        }

        int
        coherency_black_height_k(Node * node, int k) const
        {
            // when cut is enabled, black_height is not guaranteed
            // TODO: is this a problem ?
            if constexpr (CUT_ON_INSERT)
                return 1;

            if (node == nullptr)
                return 1;

            int color = (node->colors[k] == BLACK) ? 1 : 0;
            int left_child_height  = color + coherency_black_height_k(node->st[k].left,  k);
            int right_child_height = color + coherency_black_height_k(node->st[k].right, k);
            tassert(left_child_height == right_child_height);

            if (k+1 < K)
            {
                coherency_black_height_k(node->st[k+1].left,  k+1);
                coherency_black_height_k(node->st[k+1].right, k+1);
            }

            return left_child_height;
        }

        void
        coherency_black_height(Node * node) const
        {
            coherency_black_height_k(node, 0);
        }

        void
        coherency_balance(void) const
        {
            int height    = this->height();
            int size = this->size();
            int ideal_height = khp_log2(size + 1);
            if constexpr (CUT_ON_INSERT)
                tassert(height <  2 * K * ideal_height || height < 8);
            else
                tassert(height <= 2 * K * ideal_height);
        }

        void
        coherency_size(Node * node) const
        {
            int size[K];
            for (int k = 0 ; k < K ; ++k)
                size[k] = 0;
            size[node->k] = 1;

            for (int k = node->k ; k < K ; ++k)
            {
                for (int kk = 0 ; kk < K ; ++kk)
                {
                    int nl = node->st[kk].left  ? node->st[kk].left->size(k)  : 0;
                    int nr = node->st[kk].right ? node->st[kk].right->size(k) : 0;
                    size[k] += nr + nl;
                }
            }

            for (int k = 0 ; k < K ; ++k)
                tassert(size[k] == node->size(k));
        }

        void
        coherency_k_hierarchy(Node * node) const
        {
            FOREACH_CHILD_BEGIN(node, child, k, dir)
            {
                tassert(child->k == k);
                tassert(node->k <= child->k);
                tassert(k == 0 || node->hyperrect[k-1] == child->hyperrect[k-1]);
                coherency_k_hierarchy(child);
            }
            FOREACH_CHILD_END(node, child, k, dir);
        }

        void
        coherency_from(Node * root) const
        {
            /* 2. check per-node size */
            coherency_size(root);

            /* 3. If a node is red, then both its children are black */
            coherency_color(root);

            /* 4. Every path from a node to any of its descbant NULL nodes
             * has the same number of black nodes */
            coherency_black_height(root);

            /* 5. rect must be disjoint (weak check) */
            coherency_hyperrect_disjoint(root);

            /* 6. check includes.hyperrect */
            coherency_hyperrect_includes(root);

            /* 7. includeness relationship between nodes dimension */
            coherency_k(root);

            /* 8. graph must be a tree (only 1 path from root to each node) */
            coherency_single_path(root);

            /* 9. children for a k-tree must have the same k-interval as their parent */
            coherency_k_hierarchy(root);
        }

        void
        coherency_hyperrect_represented_check(Node * node, void * args) const
        {
            const Hyperrect * rect = (const Hyperrect *) args;
            tassert(rect->includes(node->hyperrect) || !rect->intersects(node->hyperrect));
        }

        // TODO : this test is incomplete, should test that all nodes union
        // also forms the rect
        void
        coherency_hyperrect_represented(Hyperrect rect)
        {
            auto f = std::bind(&LPTree<K, T>::coherency_hyperrect_represented_check, this, _1, _2);
            foreach_node(this->root, f, &rect);
        }

        int
        coherency(const Hyperrect & h)
        {
            if (this->root)
            {
                /* 1. The root of the this is always black */
                for (int k = 0 ; k < K ; ++k)
                    tassert(this->root->colors[k] == BLACK);

                /* per-node checks */
                this->coherency_from(this->root);

                /* ensure the rect is represented in the tree */
                this->coherency_hyperrect_represented(h);

                /* 7. check balance */
                this->coherency_balance();
            }

            return 1;
        }

#endif /* LP_TREE_ENABLE_COHERENCY_CHECKS */

    public:

        /////////////////////////
        // ABSTRACT INTERFACES //
        /////////////////////////

        /* called to create a new node with a rect that never appeared before. */
        virtual Node *
        new_node(
            T & t,
            const Hyperrect & h,
            const int k,
            const Color color
        ) const = 0;

        /* called to create a new node, that intersect with a previously insert
         * node 'inherit' */
        virtual Node *
        new_node(
            T & t,
            const Hyperrect & h,
            const int k,
            const Color color,
            const Node * inherit
        ) const = 0;

        # if 0
        /* called when the node being inserting includes all the descendent tree
         * return true if the subtree should be cut, false otherwise (in such
         * case, the insertion is propagated to all the subtree */
        virtual bool should_cut(T & t, Hyperrect & h, Node * parent, int k) const = 0;
        # endif

        /* called whenever this node is added to the tree */
        virtual void on_insert(Node * node, T & t) = 0;

        /* called whenever this node is being shrinked on dimension 'k' to 'interval' */
        virtual void on_shrink(Node * node, const Interval & interval, int k) = 0;

        /* called to detect whether the intersect on 'rect' should stop on the node 'node' */
        virtual bool intersect_stop_test(Node * node, T & t, const Hyperrect & h) const = 0;

        /* called whenever 'node' intersects with 'rect' */
        virtual void on_intersect(Node * node, T & t, const Hyperrect & h) const = 0;
};

#endif /* __LP_TREE_H__ */
