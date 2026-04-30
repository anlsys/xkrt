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

# include <xkrt/runtime.h>
# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>

XKRT_NAMESPACE_USE;

static int N = 0;
# define CUTOFF_DEPTH 10

static const int fib_values[] = {
    1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610,
    987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025,
    121393, 196418, 317811, 514229, 832040, 1346269, 2178309, 3524578,
    5702887, 9227465, 14930352, 24157817, 39088169, 63245986, 102334155,
    165580141, 267914296, 433494437, 701408733, 1134903170, 1836311903
};

static runtime_t runtime;

static inline int
fib(int n, int depth = 0)
{
    if (n <= 2)
        return n;

    int fn1, fn2;
    if (depth >= CUTOFF_DEPTH)
    {
        fn1 = fib(n-1, depth+1);
        fn2 = fib(n-2, depth+1);
    }
    else
    {
        runtime.task_spawn(
            [&n, &fn1, depth] (runtime_t * runtime, device_t * device, task_t * task) {
                fn1 = fib(n - 1, depth + 1);
            }
        );

        runtime.task_spawn(
            [&n, &fn2, depth] (runtime_t * runtime, device_t * device, task_t * task) {
                fn2 = fib(n - 2, depth + 1);
            }
        );
        runtime.task_wait();
    }
    return fn1 + fn2;
}

static void *
main_team(runtime_t * rt, team_t * team, thread_t * thread)
{
    # if 0
    // warmup
    if (thread->tid == 0)
    {
        fib(5);
        runtime.task_wait();
    }
    runtime.team_barrier<true>(team, thread);
    # endif

    // run
    if (thread->tid == 0)
    {
        double t0 = get_nanotime();
        int r = fib(N);
        runtime.task_wait();
        double tf = get_nanotime();
        LOGGER_INFO("Fib(%d) = %d - took %.2lf s", N, r, (tf - t0) / (double)1e9);
        assert(r == fib_values[N]);
    }
    runtime.team_barrier<true>(team, thread);

    return NULL;
}

int
main(int argc, char ** argv)
{
    if (argc != 2)
    {
        LOGGER_WARN("usage: %s [n]", argv[0]);
        N = 34;
    }
    else
        N = atoi(argv[1]);

    assert(N < sizeof(fib_values) / sizeof(int));

    assert(runtime.init() == 0);

    team_t team;
    team.desc.nthreads = 1;
    team.desc.routine = (team_routine_t) main_team;
    team.desc.master_is_member = false;

    runtime.team_create(&team);
    runtime.team_join(&team);

    assert(runtime.deinit() == 0);

    return 0;
}
