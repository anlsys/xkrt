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
 *  A moldable task is split (in halves) as long as its split-condition holds.
 *  The existing `moldability.cc` test uses a condition that never actually
 *  triggers a split; this test forces real recursive splitting and verifies:
 *
 *    1. the task was effectively split (more than one leaf executed), and
 *    2. the produced sub-intervals tile the original interval exactly once
 *       (every byte is written by exactly one leaf task).
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <atomic>
# include <stdlib.h>
# include <string.h>

XKRT_NAMESPACE_USE;

static const size_t SIZE  = 4096;   // power of two -> clean halving
static const size_t CHUNK =  512;   // stop splitting at this granularity

static std::atomic<int> nleaves;

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    unsigned char * buffer = (unsigned char *) malloc(SIZE);
    assert(buffer);
    memset(buffer, 0, SIZE);

    const uintptr_t base = (uintptr_t) buffer;
    nleaves = 0;

    runtime.task_spawn<1>(

        // accesses: a single write over the whole buffer
        [base] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, base, base + SIZE, ACCESS_MODE_W);
        },

        // split condition: keep halving while the piece is larger than CHUNK
        [] (task_t * task, access_t * accesses) {
            (void) task;
            return accesses[0].region.interval.segment[0].length() > CHUNK;
        },

        // routine: each leaf writes its (disjoint) sub-interval
        [] (runtime_t *, device_t *, task_t * task) {
            access_t * accesses = TASK_ACCESSES(task);
            const uintptr_t a = accesses[0].region.interval.segment[0].a;
            const uintptr_t b = accesses[0].region.interval.segment[0].b;

            assert(a < b);
            assert((b - a) <= CHUNK);   // splitting stopped at the granularity

            unsigned char * p = (unsigned char *) a;
            const size_t n = (size_t) (b - a);
            for (size_t i = 0 ; i < n ; ++i)
                ++p[i];

            nleaves.fetch_add(1, std::memory_order_relaxed);
        }
    );

    runtime.task_wait();

    // the task must have been split into several leaves
    assert(nleaves > 1);
    assert(nleaves == (int) (SIZE / CHUNK));

    // every byte must have been covered exactly once
    for (size_t i = 0 ; i < SIZE ; ++i)
        assert(buffer[i] == 1);

    free(buffer);

    assert(runtime.deinit() == 0);

    return 0;
}
