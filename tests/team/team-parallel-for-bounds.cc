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

/**
 *  team_parallel_for: each iteration of the index space must be executed
 *  exactly once across the team threads. We verify full, disjoint coverage
 *  for several (low, up, incr) configurations and for the templated form.
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <atomic>

XKRT_NAMESPACE_USE;

static const int MAXN = 1000;
static std::atomic<int> visits[MAXN + 1];

static void
reset(void)
{
    for (int i = 0 ; i <= MAXN ; ++i)
        visits[i].store(0, std::memory_order_relaxed);
}

/* assert visits[i]==1 for i in the arithmetic range, 0 elsewhere */
static void
check_range(int low, int up, int incr)
{
    for (int i = 0 ; i <= MAXN ; ++i)
    {
        bool in_range = false;
        if (incr > 0)
            for (int k = low ; k <= up ; k += incr)
                if (k == i) { in_range = true; break; }
        const int v = visits[i].load(std::memory_order_relaxed);
        assert(v == (in_range ? 1 : 0));
    }
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    team_t team;
    team.desc.routine = XKRT_TEAM_ROUTINE_PARALLEL_FOR;
    runtime.team_create(&team);

    auto visit = [] (thread_t * thread, const int i) {
        (void) thread;
        assert(i >= 0 && i <= MAXN);
        visits[i].fetch_add(1, std::memory_order_relaxed);
    };

    // 1. contiguous [0, MAXN], step 1
    reset();
    runtime.team_parallel_for(&team, visit, /* up */ MAXN, /* low */ 0, /* incr */ 1);
    check_range(0, MAXN, 1);

    // 2. offset start, step 3 -> [10, 900] by 3
    reset();
    runtime.team_parallel_for(&team, visit, /* up */ 900, /* low */ 10, /* incr */ 3);
    check_range(10, 900, 3);

    // 3. tiny range (fewer iterations than threads)
    reset();
    runtime.team_parallel_for(&team, visit, /* up */ 2, /* low */ 0, /* incr */ 1);
    check_range(0, 2, 1);

    // 4. templated form: <UP=100, LOW=0, INCR=1> -> [0, 100]
    reset();
    runtime.team_parallel_for<100>(&team, visit);
    check_range(0, 100, 1);

    runtime.team_join(&team);

    assert(runtime.deinit() == 0);

    return 0;
}
