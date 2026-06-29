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
 *  Exercises the access *concurrency* qualifiers, which are otherwise only
 *  set to ACCESS_CONCURRENCY_SEQUENTIAL across the test-suite:
 *
 *    - ACCESS_CONCURRENCY_COMMUTATIVE : writers run under mutual exclusion
 *      (in any order). We rely on a NON-atomic counter: lost updates would
 *      reveal a broken mutual-exclusion guarantee.
 *
 *    - ACCESS_CONCURRENCY_CONCURRENT : writers may run simultaneously. We
 *      use an atomic counter, and check that a sequential reader inserted
 *      afterwards observes every contribution.
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <atomic>
# include <unistd.h>

XKRT_NAMESPACE_USE;

/* ------------------------------------------------------------------ */
/* commutative writers -> mutual exclusion, any order                  */
/* ------------------------------------------------------------------ */
static void
test_commutative(runtime_t & runtime)
{
    LOGGER_INFO("  test_commutative");

    static const int N = 64;
    static int counter = 0;
    counter = 0;

    for (int i = 0 ; i < N ; ++i)
    {
        runtime.task_spawn<1>(
            [] (task_t * task, access_t * accesses) {
                new (accesses + 0) access_t(task,
                        (const void *) &counter,
                        ACCESS_MODE_W,
                        ACCESS_CONCURRENCY_COMMUTATIVE);
            },
            [] (runtime_t *, device_t *, task_t *) {
                /* read-modify-write WITHOUT atomics: only correct if the
                 * runtime serializes commutative writers w.r.t each other */
                int v = counter;
                usleep(50);
                counter = v + 1;
            }
        );
    }

    runtime.task_wait();
    assert(counter == N);
}

/* ------------------------------------------------------------------ */
/* concurrent writers -> may run in parallel, reader sees them all     */
/* ------------------------------------------------------------------ */
static void
test_concurrent(runtime_t & runtime)
{
    LOGGER_INFO("  test_concurrent");

    static const int N = 64;
    static std::atomic<int> counter;
    counter = 0;

    for (int i = 0 ; i < N ; ++i)
    {
        runtime.task_spawn<1>(
            [] (task_t * task, access_t * accesses) {
                new (accesses + 0) access_t(task,
                        (const void *) &counter,
                        ACCESS_MODE_W,
                        ACCESS_CONCURRENCY_CONCURRENT);
            },
            [] (runtime_t *, device_t *, task_t *) {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        );
    }

    /* a sequential reader depends on every concurrent writer */
    runtime.task_spawn<1>(
        [] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task,
                    (const void *) &counter,
                    ACCESS_MODE_R,
                    ACCESS_CONCURRENCY_SEQUENTIAL);
        },
        [] (runtime_t *, device_t *, task_t *) {
            assert(counter.load(std::memory_order_relaxed) == N);
        }
    );

    runtime.task_wait();
    assert(counter == N);
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    test_commutative(runtime);
    test_concurrent(runtime);

    assert(runtime.deinit() == 0);

    return 0;
}
