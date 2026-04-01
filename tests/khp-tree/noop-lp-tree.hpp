# include <xkrt/memory/access/common/lp-tree.hpp>
# define unused_type_t  int

static int next_id = 0;

template <int K>
class NoopLPTreeNode : public LPTree<K, unused_type_t>::Node
{
    public:
        using TreeBase = LPTree<K, unused_type_t>;
        using NodeBase = typename TreeBase::Node;

    public:
        int id;

    public:
        NoopLPTreeNode(
            const Hyperrect & r,
            const int k,
            const Color color
        ) :
            NodeBase(r, k, color),
            id(next_id++)
        {}

        void
        dump_str(FILE * f) const
        {
            fprintf(f, "%d", this->id);
        }

        void
        dump_hyperrect_str(FILE * f) const
        {
            fprintf(f, "%d", this->id);
        }

};

template<int K>
class NoopLPTree : public LPTree<K, unused_type_t>
{
    public:
        using Base      = LPTree<K, unused_type_t>;
        using Hyperrect = KHyperrect<K>;
        using Node      = NoopLPTreeNode<K>;
        using NodeBase  = typename Base::Node;
        using Search    = unused_type_t;

        Node *
        new_node(
            unused_type_t & search,
            const Hyperrect & h,
            const int k,
            const Color color
        ) const {
            return new Node(h, k, color);
        }

        Node *
        new_node(
            unused_type_t & search,
            const Hyperrect & h,
            const int k,
            const Color color,
            const NodeBase * inherit
        ) const {
           return new Node(h, k, color);
        }

        bool
        should_cut(unused_type_t & t, Hyperrect & h, NodeBase * parent, int k) const
        {
            return false;
        }

        void
        on_insert(NodeBase * node, unused_type_t & t)
        {
        }

        void
        on_shrink(NodeBase * node, const Interval & interval, int k)
        {
        }

        bool
        intersect_stop_test(NodeBase * node, unused_type_t & t, const Hyperrect & h) const
        {
            return false;
        }

        void
        on_intersect(NodeBase * node, unused_type_t & t, const Hyperrect & h) const
        {
        }
};
