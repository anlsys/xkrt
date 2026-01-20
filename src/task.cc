/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
** Joao Lima joao.lima@inf.ufsm.br
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
# include <xkrt/internals.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/bits-to-str.h>
# include <xkrt/logger/todo.h>
# include <xkrt/sync/mem.h>
# include <xkrt/stats/stats.h>
# include <xkrt/task/task.hpp>

# include <cassert>
# include <cstring>
# include <cerrno>

XKRT_NAMESPACE_BEGIN

static inline task_format_target_t
driver_type_to_task_format_target(driver_type_t driver_type)
{
    switch (driver_type)
    {
        # define CASE(X)                        \
            case (XKRT_DRIVER_TYPE_##X):        \
                return XKRT_TASK_FORMAT_TARGET_##X;  \
                break ;

        CASE(HOST)
        CASE(CUDA)
        CASE(ZE)
        CASE(CL)
        CASE(HIP)
        CASE(SYCL)

        # undef CASE

        default:
            LOGGER_FATAL("Invalid device driver type");
    }
}

static inline driver_type_t
task_format_target_to_driver_type(task_format_target_t fmt)
{
    switch (fmt)
    {
        # define CASE(X)                        \
            case (XKRT_TASK_FORMAT_TARGET_##X):      \
                return XKRT_DRIVER_TYPE_##X;    \
                break ;

        CASE(HOST)
        CASE(CUDA)
        CASE(ZE)
        CASE(CL)
        CASE(HIP)
        CASE(SYCL)

        # undef CASE

            default:
            LOGGER_FATAL("Invalid task format target");
    }
}

static inline
device_global_id_t
__task_guess_device(
    runtime_t * runtime,
    task_t * task
) {
    // if that task must execute on a device
    if (task->flags & TASK_FLAG_DEVICE)
    {
        if (task->fmtid != XKRT_TASK_FORMAT_NULL)
        {
            task_format_t * format = runtime->task_format_get(task->fmtid);
            if (format->suggest)
                LOGGER_FATAL("Prefetch not supported if a suggested device is specified");
        }

        task_dev_info_t * dev = TASK_DEV_INFO(task);
        assert(dev);

        // if it has a targeted device set, return it
        if (dev->targeted_device_id != UNSPECIFIED_DEVICE_GLOBAL_ID)
            return dev->targeted_device_id;

        // if it has an OCR, we may already know which device will own most
        // bytes once all predecessor tasks finished their execution, but its
        // not trivial to implement at that point. So it is not supported yet.
        // This should:
        //  - for each access of the task
        //      - get predecessor writers
        //      - get place of execution of writers
        //      - if they are all known,
        //          - return the device with most bytes written
        //      - else
        //          - if amongst known devices, one owns more than half of the bytes,
        //              - return this device
        //          - else
        //              - we dont know yet, return UNSPECIFIED_DEVICE
        if (dev->ocr_access_index)
        {
            # pragma message(TODO "See file and comment: implement pre-fetching for tasks with OCR")
        }

        // could not find the device to execute
        return UNSPECIFIED_DEVICE_GLOBAL_ID;
    }
    // else, it executes on the host
    else
        return HOST_DEVICE_GLOBAL_ID;
}

/**
 *  - transition the task to completed
 *  - initiate memory prefetching for successors whose place of execution is known
 *  - enqueue all ready successors
 */
static inline void
__task_complete(
    runtime_t * runtime,
    task_t * task
) {
    // assertions
    assert(
        task->state.value == TASK_STATE_DATA_FETCHED    ||
        task->state.value == TASK_STATE_EXECUTING       ||
        task->state.value == TASK_STATE_READY
    );
    if (task->flags & TASK_FLAG_DEPENDENT)
        assert(TASK_DEP_INFO(task)->wc.load() == 0);
    if (task->flags & TASK_FLAG_DETACHABLE)
        assert(TASK_DET_INFO(task)->wc.load() == 0);

    // transition the task
    SPINLOCK_LOCK(task->state.lock);
    {
        task->state.value = TASK_STATE_COMPLETED;
        LOGGER_DEBUG_TASK_STATE(task);
    }
    SPINLOCK_UNLOCK(task->state.lock);
    assert(task->parent);

    // TODO: instead, can we have a counter per thread, to reduce the number of
    // updates on the 'parent' counter ?
    XKRT_STATS_INCR(runtime->stats.tasks[task->fmtid].completed, 1);
    task->parent->cc.fetch_sub(1, std::memory_order_relaxed);

    // if the task has successors, that dependency is now satisfied
    if (task->flags & TASK_FLAG_DEPENDENT)
    {
        task_dep_info_t * dep = TASK_DEP_INFO(task);
        access_t * accesses = TASK_ACCESSES(task);
        for (task_access_counter_t i = 0 ; i < dep->ac ; ++i)
        {
            access_t * access = accesses + i;

            // detached access, not my responsibility to fulfill this dependency
            if (access->mode & ACCESS_MODE_D)
                continue ;

            for (access_t * succ_access : access->successors)
            {
                // get successor task
                task_t * succ = succ_access->task;

                ////////////////////////
                // MEMORY PREFETCHING //
                ////////////////////////

                if (runtime->conf.enable_prefetching)
                {
                    // if the succ access is not being fetched, or got fetched already
                    if (succ_access->state == ACCESS_STATE_INIT)
                    {
                        // if the pred access wrote memory
                        if (access->mode & ACCESS_MODE_W)
                        {
                            // if successor device can already be known
                            const device_global_id_t device_global_id = __task_guess_device(runtime, succ);
                            if (device_global_id != UNSPECIFIED_DEVICE_GLOBAL_ID)
                            {
                                // then we can prefetch memory
                                MemoryCoherencyController * mcc = task_get_memory_controller(runtime, succ->parent, succ_access);
                                if (mcc)
                                    mcc->fetch(succ_access, device_global_id);
                            }
                        }
                    }
                }

                //////////////////////////////////
                // RELEASE TASK DEPENPENDENCIES //
                //////////////////////////////////

                assert(succ->flags & TASK_FLAG_DEPENDENT);
                task_dep_info_t * sdep = TASK_DEP_INFO(succ);

                // task may be ready now
                if (sdep->wc.fetch_sub(1, std::memory_order_seq_cst) == 1)
                    __task_ready(succ, runtime_submit_task, runtime);
            }
        }
    }
}

/* decrease detachable ref counter by 1, and complete the task if it reached 0 */
template <int N>
static inline void
__task_detachable_decr(
    runtime_t * runtime,
    task_t * task
) {
    assert(task->flags & TASK_FLAG_DETACHABLE);
    task_det_info_t * det = TASK_DET_INFO(task);
    if (det->wc.fetch_sub(N, std::memory_order_relaxed) == N)
        __task_complete(runtime, task);
}

/* increase detachable ref counter by 1 */
template <int N>
static inline void
__task_detachable_incr(
    task_t * task
) {
    assert(task->flags & TASK_FLAG_DETACHABLE);
    task_det_info_t * det = TASK_DET_INFO(task);
    det->wc.fetch_add(N, std::memory_order_relaxed);
}

/* transition the task to the state 'executed' - and eventually to 'completed'
 * or 'detached' */
static inline void
__task_executed(
    runtime_t * runtime,
    task_t * task
) {
    assert(task->state.value == TASK_STATE_EXECUTING);

    if (task->flags & TASK_FLAG_DETACHABLE)
        __task_detachable_decr<1>(runtime, task);
    else
        __task_complete(runtime, task);
}

/**
 *  Execute a task.  * Must be called once all task accesses were fetched.
 */
void
task_execute(runtime_t * runtime, device_t * device, task_t * task)
{
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task->state.value = TASK_STATE_EXECUTING;
    LOGGER_DEBUG_TASK_STATE(task);

    // if detachable, increase counter to avoid early completion (before routine executed)
    if (task->flags & TASK_FLAG_DETACHABLE)
        __task_detachable_incr<1>(task);

    task_format_t * format;

    /* running an empty task */
    if (task->fmtid == XKRT_TASK_FORMAT_NULL)
    {
        __task_executed(runtime, task);
    }
    else
    {
        /* retrieve task format */
        format = runtime->task_format_get(task->fmtid);
        assert(format);

       /* if there is a format */
        if (format)
        {
            task_format_target_t targetfmt;
            if (device)
            {
                targetfmt = driver_type_to_task_format_target(device->driver_type);
                if (format->f[targetfmt] == NULL)
                    targetfmt = XKRT_TASK_FORMAT_TARGET_HOST;
            }
            else
                targetfmt = XKRT_TASK_FORMAT_TARGET_HOST;

            if (format->f[targetfmt])
            {
                task_t * current = thread->current_task;
                thread->current_task = task;
                ((void (*)(runtime_t *, device_t *, task_t *)) format->f[targetfmt])(runtime, device, task);
                thread->current_task = current;

                /* if the task yielded, requeue it */
                if (task->flags & TASK_FLAG_REQUEUE)
                {
                    task->flags = task->flags & ~(TASK_FLAG_REQUEUE);
                    runtime->task_thread_enqueue(thread, task);
                }
                /* else, it executed entirely */
                else
                    __task_executed(runtime, task);
            }
            else
                LOGGER_FATAL("Task format for `%p` has no impl for device `%u`", task, device->global_id);
        }
        else
            LOGGER_FATAL("Invalid format for task `%p`", task);
    }
}

static void
body_host_capture(runtime_t * runtime, device_t * device, task_t * task)
{
    assert(task);

    std::function<void(runtime_t *, device_t *, task_t *)> * f =
        (std::function<void(runtime_t *, device_t *, task_t *)> *) TASK_ARGS(task);
    (*f)(runtime, device, task);
}

void
task_host_capture_register_format(runtime_t * runtime)
{
    task_format_t format;
    memset(format.f, 0, sizeof(format.f));
    format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_host_capture;
    snprintf(format.label, sizeof(format.label), "host_capture");
    runtime->formats.host_capture = runtime->task_format_create(&format);
}

void
runtime_t::task_commit(task_t * task)
{
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    thread->commit(task, runtime_submit_task, this);
    XKRT_STATS_INCR(this->stats.tasks[task->fmtid].commited, 1);
}

void
runtime_t::task_detachable_decr(task_t * task)
{
    assert(task);
    assert(task->flags & TASK_FLAG_DETACHABLE);
    assert(task->state.value != TASK_STATE_COMPLETED);
    __task_detachable_decr<1>(this, task);
}

void
runtime_t::task_detachable_incr(task_t * task)
{
    assert(task);
    assert(task->flags & TASK_FLAG_DETACHABLE);
    assert(task->state.value != TASK_STATE_COMPLETED);
    __task_detachable_incr<1>(task);
}

void
runtime_t::task_complete(task_t * task)
{
    assert(task);
    assert(!(task->flags & TASK_FLAG_DETACHABLE));

    __task_complete(this, task);
}

/** duplicate a moldable task */
task_t *
runtime_t::task_dup(
    const task_t * task
) {
    assert(task->flags & TASK_FLAG_MOLDABLE);

    task_mol_info_t * mol = TASK_MOL_INFO(task);
    assert(mol);

    const size_t args_size = mol->args_size;
    const size_t task_size = TASK_SIZE(task);

    task_t * dup = this->task_allocate<false>(task_size + args_size);

    // TODO: probably not C++ standard, but should work ?
    memcpy(dup, task, task_size + args_size);

    # if XKRT_SUPPORT_DEBUG
    snprintf(dup->label, sizeof(dup->label), "%s-dup", task->label);
    # endif

    return dup;
}

/////////////////////
// TASK SUBMISSION //
/////////////////////

static inline void
submit_task_host(
    runtime_t * runtime,
    task_t * task
) {
    thread_t * tls = thread_t::get_tls();
    assert(tls);

    // tls->team == NULL means it come from a user-thread, unknown to kaapi
    // tls->device_global_id != XKRT_DRIVER_TYPE_HOST means it is a kaapi thread, but not a host thread
    if (tls->team == NULL || tls->device_global_id != HOST_DEVICE_GLOBAL_ID)
    {
        device_t * device = runtime->device_get(HOST_DEVICE_GLOBAL_ID);
        runtime->task_team_enqueue(device->team, task);
    }
    else
    {
        assert(tls->team == runtime->device_get(HOST_DEVICE_GLOBAL_ID)->team);
        runtime->task_thread_enqueue(tls, task);
    }
}

static inline void
submit_task_device(
    runtime_t * runtime,
    task_t * task
) {
    // task must be flagged
    assert(task->flags & TASK_FLAG_DEVICE);
    task_dev_info_t * dev = TASK_DEV_INFO(task);

    // Bitfield of devices eligible to the task
    // Initially set to all devices, logic bellow removes bits from it to elect only one
    device_global_id_bitfield_t devices_bitfield = XKRT_DEVICES_MASK_ALL;

    // if the task format suggests a format type, filter for the devices that supports this format
    task_format_t * format;
    if (task->fmtid != XKRT_TASK_FORMAT_NULL)
    {
        format = runtime->task_format_get(task->fmtid);
        assert(format);
        if (format->suggest)
        {
            task_format_target_t fmt_target = format->suggest(task);
            if (fmt_target != XKRT_TASK_FORMAT_TARGET_NO_SUGGEST)
            {
                driver_type_t driver_type = task_format_target_to_driver_type(fmt_target);
                device_global_id_bitfield_t suggested_devices = runtime->devices_get(driver_type);
                if (devices_bitfield & suggested_devices)
                    devices_bitfield &= suggested_devices;
            }
        }
    }

    // if an owner-computes rules (ocr) parameter is set, filter to keep only
    // devices that owns the larger volume of coherent bytes
    if (dev->ocr_access_index != UNSPECIFIED_TASK_ACCESS)
    {
        // if an ocr is set, task must be a dependent task (i.e. with some accesses)
        assert(task->flags & TASK_FLAG_DEPENDENT);

        // retrieve the access
        task_dep_info_t * dep = TASK_DEP_INFO(task);
        assert(dev->ocr_access_index >= 0 && dev->ocr_access_index < dep->ac);
        access_t * access = TASK_ACCESSES(task) + dev->ocr_access_index;

        // looking for the device that owns the data
        MemoryCoherencyController * memcontroller = task_get_memory_controller(runtime, task->parent, access);
        assert(memcontroller);

        // retrieve owners excluding the host device
        const device_global_id_bitfield_t owners = memcontroller->who_owns(access) & ~(1 << HOST_DEVICE_GLOBAL_ID);;

        // if there is no owners in the eligible list
        if ((devices_bitfield & owners) == 0)
        {
            // keep all devices eligible
        }
        else
        {
            // keep only owners
            devices_bitfield &= owners;
        }

        assert(devices_bitfield);
    }

    // programmer provided an explicit targeted device
    if (dev->targeted_device_id != UNSPECIFIED_DEVICE_GLOBAL_ID)
    {
        // it is present in the bitfield, then select that device
        if (devices_bitfield & (1 << dev->targeted_device_id))
            devices_bitfield = (device_global_id_bitfield_t) (1 << dev->targeted_device_id);
    }

    //////////////////////////////////////

    // At that point, 'device_bitfield' contains the list of eligible devices
    // We retrieve only one now

    assert(devices_bitfield);
    device_global_id_t device_global_id = HOST_DEVICE_GLOBAL_ID;

    // if any device available, pick one
    if (devices_bitfield != (1 << HOST_DEVICE_GLOBAL_ID))
    {
        // bitmask of all devices but the host
        device_global_id_bitfield_t bitmask = (device_global_id_bitfield_t) ((1 << runtime->drivers.devices.n) - 1) & ~(1 << HOST_DEVICE_GLOBAL_ID);

        // pick randomly
        device_global_id = (device_global_id_t) __random_set_bit(devices_bitfield & bitmask) - 1;
    }

    // save device id into the task info
    assert((device_global_id >= 0 && device_global_id < runtime->drivers.devices.n));
    dev->elected_device_id = device_global_id;

    LOGGER_DEBUG("Enqueuing task `%s` to device %d", task->label, device_global_id);

    device_t * device = runtime->drivers.devices.list[device_global_id];
    assert(device);

    /* push a task to a thread of the device */
    runtime->task_team_enqueue(device->team, task);
}

/**
 *  Entry point when a task is ready to be fetched.
 *  It elects a thread and a device for fetching accesses and executing the task
 */
void
runtime_submit_task(
    runtime_t * runtime,
    task_t * task
) {
    assert(task->state.value == TASK_STATE_READY);

    /* if the task is flagged, then schedule it onto an implicit team of threads */
    if (task->flags & TASK_FLAG_DEVICE)
        submit_task_device(runtime, task);
    /* else, submit it whether to the host implicit team, or to the currently executing team */
    else
        submit_task_host(runtime, task);
}

void
runtime_t::task_enqueue(task_t * task)
{
    runtime_submit_task(this, task);
}

XKRT_NAMESPACE_END
