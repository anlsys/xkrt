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
 *  Verifies that data-dependencies are honored when tasks are spawned through
 *  the high-level capture API `task_spawn<ac>(set_accesses, routine)`.
 *
 *  Sub-tests:
 *    1. RAW / WAR / WAW chain on a single point -> strict serialization
 *    2. fan-out  (1 writer  -> N concurrent readers)
 *    3. fan-in   (N serialized writers -> 1 reader)
 *
 *  Each writer sleeps briefly before publishing its value: if a dependency
 *  was NOT enforced, a successor would observe a stale value and assert.
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <atomic>
# include <stdlib.h>
# include <string.h>
# include <unistd.h>

XKRT_NAMESPACE_USE;

/* ------------------------------------------------------------------ */
/* Test 1: RAW/WAR/WAW chain on a single point => strict order         */
/* ------------------------------------------------------------------ */
static void
test_chain(runtime_t & runtime)
{
    LOGGER_INFO("  test_chain (RAW/WAR/WAW)");

    static int value = 0;
    value = 0;

    /* T0: W  -> value = 1 */
    runtime.task_spawn<1>(
        [] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, (const void *) &value, ACCESS_MODE_W);
        },
        [] (runtime_t *, device_t *, task_t *) {
            usleep(2000);
            value = 1;
        }
    );

    /* T1: R  -> must observe 1 (RAW on T0) */
    runtime.task_spawn<1>(
        [] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, (const void *) &value, ACCESS_MODE_R);
        },
        [] (runtime_t *, device_t *, task_t *) {
            assert(value == 1);
        }
    );

    /* T2: W  -> WAR on T1, WAW on T0; sets value = 2 */
    runtime.task_spawn<1>(
        [] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, (const void *) &value, ACCESS_MODE_W);
        },
        [] (runtime_t *, device_t *, task_t *) {
            assert(value == 1);
            usleep(2000);
            value = 2;
        }
    );

    /* T3: RW -> RAW+WAW on T2; sets value = 3 */
    runtime.task_spawn<1>(
        [] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, (const void *) &value, ACCESS_MODE_RW);
        },
        [] (runtime_t *, device_t *, task_t *) {
            assert(value == 2);
            value = 3;
        }
    );

    runtime.task_wait();
    assert(value == 3);
}

/* ------------------------------------------------------------------ */
/* Test 2: fan-out (1 writer -> N readers)                             */
/* ------------------------------------------------------------------ */
static void
test_fanout(runtime_t & runtime)
{
    LOGGER_INFO("  test_fanout");

    static const int   N    = 16;
    static const size_t size = 256;
    unsigned char * buffer = (unsigned char *) malloc(size);
    assert(buffer);
    memset(buffer, 0, size);

    const uintptr_t a = (uintptr_t) buffer;
    const uintptr_t b = a + size;

    static std::atomic<int> readers_done;
    readers_done = 0;

    /* writer: fill the buffer */
    runtime.task_spawn<1>(
        [a, b] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, a, b, ACCESS_MODE_W);
        },
        [buffer] (runtime_t *, device_t *, task_t *) {
            usleep(2000);
            for (size_t i = 0 ; i < size ; ++i)
                buffer[i] = (unsigned char) (i % 256);
        }
    );

    /* N readers: each must see the fully written buffer */
    for (int r = 0 ; r < N ; ++r)
    {
        runtime.task_spawn<1>(
            [a, b] (task_t * task, access_t * accesses) {
                new (accesses + 0) access_t(task, a, b, ACCESS_MODE_R);
            },
            [buffer] (runtime_t *, device_t *, task_t *) {
                for (size_t i = 0 ; i < size ; ++i)
                    assert(buffer[i] == (unsigned char) (i % 256));
                ++readers_done;
            }
        );
    }

    runtime.task_wait();
    assert(readers_done == N);

    free(buffer);
}

/* ------------------------------------------------------------------ */
/* Test 3: fan-in (N serialized writers -> 1 reader)                   */
/* ------------------------------------------------------------------ */
static void
test_fanin(runtime_t & runtime)
{
    LOGGER_INFO("  test_fanin");

    static const int N = 32;
    static int counter = 0;
    counter = 0;

    /* N writers on the same point: WAW serializes them, each does +1 */
    for (int i = 0 ; i < N ; ++i)
    {
        runtime.task_spawn<1>(
            [] (task_t * task, access_t * accesses) {
                new (accesses + 0) access_t(task, (const void *) &counter, ACCESS_MODE_RW);
            },
            [] (runtime_t *, device_t *, task_t *) {
                /* no atomics: correctness relies on serialization */
                int v = counter;
                counter = v + 1;
            }
        );
    }

    /* reader: RAW on the last writer */
    runtime.task_spawn<1>(
        [] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, (const void *) &counter, ACCESS_MODE_R);
        },
        [] (runtime_t *, device_t *, task_t *) {
            assert(counter == N);
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

    test_chain(runtime);
    test_fanout(runtime);
    test_fanin(runtime);

    assert(runtime.deinit() == 0);

    return 0;
}
