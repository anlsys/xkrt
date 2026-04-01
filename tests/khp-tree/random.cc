# include "noop-lp-tree.hpp"

# include <math.h>
# include <stdlib.h>
# include <stdint.h>
# include <time.h>

uint64_t
get_nanotime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)(ts.tv_sec * 1000000000) + (uint64_t) ts.tv_nsec;
}

// number of insertion counter
static int ninsert = 0;

// Insert a region into the tree
template<int K>
void
insert(
    NoopLPTree<K> & tree,
    Interval intervals[K]
) {
    KHyperrect<K> h(intervals);
    unused_type_t t;
    tree.insert(t, h);
    ++ninsert;

    # if 0
    std::cout << "Exporting pdf..." << std::endl;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "random-k-%d-%d", K, ninsert);
    tree.export_pdf(buffer);
    # endif

}

//Generate 'n' disjoint hyperplans
template<int K, int PLAN_DIM>
void
disjoint_hyperplans(NoopLPTree<K> & tree, int n)
{
    static_assert(0 <= PLAN_DIM && PLAN_DIM < K);

    for (int i = 0 ; i < n ; ++i)
    {
        Interval intervals[K];
        for (int k = 0 ; k < K ; ++k)
        {
            if (k != PLAN_DIM)
            {
                intervals[k].a =   0;
                intervals[k].b = n*8;
            }
            else
            {
                intervals[k].a = 4*(i+0);
                intervals[k].b = 4*(i+1);
            }
        }
        insert(tree, intervals);
    }
}

//Generate 'n' hyperhs that includes all dimensions but one
template<int K>
void
pyramid(NoopLPTree<K> & tree, int n)
{
    for (int i = 0 ; i < n ; ++i)
    {
        Interval intervals[K];
        for (int k = 0 ; k < K ; ++k)
        {
            if (k < K - 1)
            {
                intervals[k].a = n*8/2-4*(i+1);
                intervals[k].b = n*8/2+4*(i+1);
            }
            else
            {
                intervals[k].a = 4*(i+0);
                intervals[k].b = 4*(i+1);
            }
        }
        insert(tree, intervals);
    }
}

//Generate 'n' hyperhs that are included on all dimensions but one
template<int K>
void
pyramid_inverted(NoopLPTree<K> & tree, int n)
{
    for (int i = 0 ; i < n ; ++i)
    {
        Interval intervals[K];
        for (int k = 0 ; k < K ; ++k)
        {
            if (k < K - 1)
            {
                intervals[k].a = 0+4*i;
                intervals[k].b = n*8-4*i;
            }
            else
            {
                intervals[k].a = 4*(i+0);
                intervals[k].b = 4*(i+1);
            }
        }
        insert(tree, intervals);
    }
}

// Generate 'n' hyperhs that are successively included into previous ones
template<int K>
static void
squares_included(NoopLPTree<K> & tree, int n)
{
    for (int i = 0 ; i < n ; ++i)
    {
        Interval intervals[K];
        for (int k = 0 ; k < K ; ++k)
        {
            intervals[k].a =     i*4;
            intervals[k].b = n*8-i*4;
        }
        insert<K>(tree, intervals);
    }
}

template<int Dimensions, int K, class Callable>
constexpr void meta_for_loop(NoopLPTree<K> & tree, std::array<int, K> & array, int end, Callable & c)
{
    static_assert(Dimensions > 0);
    for(int i = 0; i != end; ++i)
    {
        array[Dimensions-1] = i;
        if constexpr(Dimensions > 1)
            meta_for_loop<Dimensions-1, K>(tree, array, end, c);
        else
            c(tree, array);
    }
}

template<int K, int P>
static void
matrix_tiles_insert(NoopLPTree<K> & tree, std::array<int, K> indices)
{
    Interval intervals[K];
    for (int k = 0 ; k < K ; ++k)
    {
        intervals[k].a = (indices[k]+0)*P;
        intervals[k].b = (indices[k]+1)*P;
    }
    insert(tree, intervals);
}

// Generate n^K tiles of size P
template<int K, int P>
static void
matrix_tiles(NoopLPTree<K> & tree, int n)
{
    std::array<int, K> array;
    meta_for_loop<K, K>(tree, array, n, matrix_tiles_insert<K, P>);
}

// Test size
static int N = 10;

// Launch tests for a tree of dimension 'K'
template<int K>
static void launch_tests(NoopLPTree<K> & tree)
{
    printf("Running for K=%d\n", K);
    ninsert = 0;

    uint64_t t0 = get_nanotime();
    {
        # if 1

        matrix_tiles<K, 15>(tree, 2*8);
        matrix_tiles<K, 10>(tree, 3*8);

        # elif 0
        for (int i = 0 ; i < N ; ++i)
        {
            Interval interval[K] = {
                { .a = 0,  .b = 16      },
                { .a = 4*i, .b = 4*(i+1) },
            };
            insert<K>(tree, interval);
        }

        int xx[] = {
            4, 1024,
            1028,2048,
            2564, 4096,
            2052,2304,
            2308,2432,
            2436,2496,
            2500, 2528,
            2532,2544,
        };

        for (unsigned int i = 0 ; i < sizeof(xx) / sizeof(int) ; i += 2)
        {
            Interval interval[K] = {
                { .a = 0,  .b = 16  },
                { .a = xx[i+0],  .b = xx[i+1] },
            };
            insert<K>(tree, interval);
        }

        # else

        // hyperplans
        disjoint_hyperplans<K, K-1>(tree, N);
        disjoint_hyperplans<K,   0>(tree, N);

        // matrix test
        int Nth_sqrt = std::pow(N, 1.0/K);
        matrix_tiles<K, 2>(tree, Nth_sqrt/2*2+1);
        matrix_tiles<K, 3>(tree, Nth_sqrt/3*2+1);
        matrix_tiles<K, 5>(tree, Nth_sqrt/5*2+1);
        matrix_tiles<K, 7>(tree, Nth_sqrt/7*2+1);

        // include squares
        squares_included<K>(tree, N);

        // pyramid inverted
        pyramid_inverted<K>(tree, N);

        // pyramid
        pyramid<K>(tree, N);

        # endif
    }
    uint64_t tf = get_nanotime();

    // finish
    double dt = (tf - t0) / 1e9;
    int height    = tree.height();
    int nelements = tree.size();
    printf("  Took %lf s.\n", dt);
    printf("    Inserted %d h and %d elements - height is %d\n", ninsert, nelements, height);
    printf("    Hyperrect/s. = %.2lf\n", ninsert / dt);
    printf("    Elements/s. = %.2lf\n", nelements / dt);
    printf("\n");
}

// Run for dimension 'K'
template<int K>
static void run(void)
{
    NoopLPTree<K> tree;
    launch_tests(tree);

    # if 1
    std::cout << "Exporting pdf..." << std::endl;
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "random-k-%d", K);
    tree.export_pdf(buffer);
    # endif
}

int
main(int argc, char ** argv)
{
    if (argc == 2)
        N = atoi(argv[1]);

    printf("Testing with N=%d, you can change running `%s [N]`\n\n", N, argv[0]);

//    run<1>();
    run<2>();
//    run<3>();
//    run<4>();
//    run<5>();
//    run<6>();

    return 0;
}
