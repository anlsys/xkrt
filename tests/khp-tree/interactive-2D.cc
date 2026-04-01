# include <iostream>
# include "noop-lp-tree.hpp"

# define K 2

int
main(void)
{
    NoopLPTree<K> tree;

    int x0, x1, y0, y1;
    std::cout << "Enter rectangles in format `x0 y0 x1 y1` or press CTRL+D to stop:" << std::endl;
    std::cout << "> ";
    while (std::cin >> x0 >> y0 >> x1 >> y1)
    {
        const Interval list[K] = {
            Interval(x0, x1),
            Interval(y0, y1)
        };
        KHyperrect<K> rect(list);
        unused_type_t t;
        tree.insert(t, rect);
        tree.export_pdf("interactive");

        std::cout << "> ";
    }

    return 0;
}
