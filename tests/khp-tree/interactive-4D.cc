# include <iostream>
# include "noop-lp-tree.hpp"

# define K 4

int
main(void)
{
    NoopLPTree<K> tree;

    int x0, x1, y0, y1, z0, z1, w0, w1;
    std::cout << "Enter rectangles in format `x0 x1 y0 y1 z0 z1 w0 w1` or press CTRL+D to stop:" << std::endl;
    std::cout << "> ";
    while (std::cin >> x0 >> x1 >> y0 >> y1 >> z0 >> z1 >> w0 >> w1)
    {
        const Interval list[K] = {
            Interval(x0, x1),
            Interval(y0, y1),
            Interval(z0, z1),
            Interval(w0, w1),
        };
        KCube<K> cube(list);
        unused_type_t t;
        tree.insert(t, cube);
        tree.export_pdf("interactive");

        std::cout << "> ";
    }

    return 0;
}
