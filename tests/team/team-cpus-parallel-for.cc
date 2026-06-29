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

# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif
# include <sched.h>

#if !XKRT_SUPPORT_DEBUG
# define XKRT_ASSERT(...) if (__VA_ARGS__) {}
#else
# define XKRT_ASSERT(...) assert(__VA_ARGS__)
#endif

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>

XKRT_NAMESPACE_USE;

int
main(void)
{
    runtime_t runtime;
    XKRT_ASSERT(runtime.init() == 0);

    // team on all cpus
    team_t team;
    team.desc.routine = XKRT_TEAM_ROUTINE_PARALLEL_FOR;

    // TEST 1
    // create and destroy the team without working
    {
        std::atomic<int> counter(0);
        runtime.team_create(&team);
        runtime.team_join(&team);
        XKRT_ASSERT(counter == 0);
    }

    // TEST 2
    // create, dispatch 1 function, destroy
    {
        std::atomic<int> counter(0);
        runtime.team_create(&team);
        runtime.team_parallel_for(&team, [&counter] (thread_t * thread) {
                LOGGER_INFO("Thread `%3d` running on `sched_getcpu() -> %3d`", thread->tid, sched_getcpu());
                ++counter;
            }
        );
        runtime.team_join(&team);
        XKRT_ASSERT(counter == team.priv.nthreads);
    }

    // TEST 3
    // create, dispatch 'n' functions, destroy
    {
        constexpr int n = 10000;
        std::atomic<int> counter(0);
        runtime.team_create(&team);

        uint64_t t0 = get_nanotime();
        for (int i = 0 ; i < n ; ++i)
            runtime.team_parallel_for(&team, [] (thread_t * thread) { });
        uint64_t tf = get_nanotime();
        LOGGER_INFO("`%d` empty parallel on `%d` threads for took %lf s - that is %luns/task\n", n, team.priv.nthreads, (tf-t0)/1e9, (tf-t0)/(n*team.priv.nthreads));

        runtime.team_join(&team);
    }

    // TEST 4
    {
        runtime.team_create(&team);
        runtime.team_parallel_for<64>(&team, [] (thread_t * thread, const int i) {
                LOGGER_INFO("Thread `%3d` running iter %d", thread->tid, i);
            }
        );
        runtime.team_join(&team);
    }

    XKRT_ASSERT(runtime.deinit() == 0);

    return 0;
}
