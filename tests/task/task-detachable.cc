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
 *  Detachable tasks (TASK_FLAG_DETACHABLE) are NOT tied to GPUs: a detachable
 *  task's *completion* is decoupled from the return of its routine and instead
 *  driven by a reference counter (task_detachable_incr / task_detachable_decr).
 *
 *  This test creates a detachable task whose routine spawns background work on
 *  another std::thread and increments the detach counter. The task is only
 *  completed once that background work decrements the counter back to 0.
 *
 *  Correctness is verified through task_wait(): it must not return until the
 *  *background* work has finished (i.e. the counter reached 0). If completion
 *  were (incorrectly) tied to the routine returning, task_wait() would return
 *  while the background thread is still sleeping and `detached_work_done` would
 *  still be false.
 *
 *  Note: the task has no successor on purpose. Completing a task that has
 *  successors enqueues them, which must happen from an XKRT worker thread; here
 *  completion is driven from a plain std::thread, which is only safe when there
 *  is nothing to enqueue.
 */

# include <xkrt/runtime.h>
# include <xkrt/task/format.h>
# include <xkrt/task/task.hpp>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <atomic>
# include <string.h>
# include <thread>
# include <unistd.h>

XKRT_NAMESPACE_USE;

static int                x                  = 0;
static std::atomic<bool>  detached_work_done(false);
static std::thread        worker;

/* detachable task: defers its completion to a background thread */
static void
detach_body(runtime_t * runtime, device_t * device, task_t * task)
{
    (void) device;

    // keep the task "pending": completion is now gated on the matching decr
    runtime->task_detachable_incr(task);

    // background work that completes the task asynchronously
    worker = std::thread([runtime, task] () {
        usleep(100 * 1000);  // 100 ms: ensure the routine returned meanwhile
        x = 42;
        detached_work_done.store(true, std::memory_order_release);
        runtime->task_detachable_decr(task);  // drops the counter to 0 -> complete
    });

    // routine returns here, but the task must NOT be considered complete yet
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    // register a host format for the detachable task
    task_format_id_t fmtid;
    {
        task_format_t format;
        memset(&format, 0, sizeof(task_format_t));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) detach_body;
        snprintf(format.label, sizeof(format.label), "detach");
        fmtid = runtime.task_format_create(&format);
    }
    assert(fmtid);

    // a single detachable task with one (write) access, no successor
    {
        constexpr task_flag_bitfield_t flags = TASK_FLAG_ACCESSES | TASK_FLAG_DETACHABLE;
        constexpr task_access_counter_t AC = 1;
        static_assert(AC <= XKRT_TASK_MAX_ACCESSES);

        task_t * task = runtime.task_new(fmtid, flags, NULL, 0, AC);

        // construct the dependency + detach info structs
        new (TASK_ACS_INFO(task)) task_acs_info_t(AC);
        new (TASK_DET_INFO(task)) task_det_info_t();   // counter starts at 0

        # if XKRT_SUPPORT_DEBUG
        snprintf(task->label, sizeof(task->label), "detachable");
        # endif

        access_t * accesses = TASK_ACCESSES(task);
        new (accesses + 0) access_t(task, (const void *) &x, ACCESS_MODE_W);
        runtime.task_accesses_resolve(accesses, AC);

        runtime.task_commit(task);
    }

    // task_wait must block until the detached background work completed the task
    runtime.task_wait();

    if (worker.joinable())
        worker.join();

    // if completion had been (wrongly) tied to the routine returning, task_wait
    // could have returned before the background thread published these
    assert(detached_work_done.load());
    assert(x == 42);

    assert(runtime.deinit() == 0);

    return 0;
}
