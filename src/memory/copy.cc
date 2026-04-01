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

XKRT_NAMESPACE_BEGIN

typedef struct  copy_args_t
{
    // the runtime
    runtime_t * runtime;

    // the device responsible to perform the copy
    device_unique_id_t device_unique_id;

    // pointers
    device_unique_id_t dst_device_unique_id;
    uintptr_t dst_device_mem;

    device_unique_id_t src_device_unique_id;
    uintptr_t src_device_mem;

    // size of the copy
    size_t size;

}               copy_args_t;

void
runtime_t::copy(
    const device_unique_id_t      device_unique_id,
    const memory_view_t         & host_view,
    const device_unique_id_t      dst_device_unique_id,
    const memory_replica_view_t & dst_device_view,
    const device_unique_id_t      src_device_unique_id,
    const memory_replica_view_t & src_device_view,
    const callback_t            & callback
) {
    device_t * device = this->device_get(device_unique_id);
    device->offloader_queue_command_submit_copy<memory_view_t, memory_replica_view_t>(
        host_view,
        dst_device_unique_id,
        dst_device_view,
        src_device_unique_id,
        src_device_view,
        callback
    );
}

void
runtime_t::copy(
    const device_unique_id_t   device_unique_id,
    const size_t               size,
    const device_unique_id_t   dst_device_unique_id,
    const uintptr_t            dst_device_addr,
    const device_unique_id_t   src_device_unique_id,
    const uintptr_t            src_device_addr,
    const callback_t         & callback
) {
    device_t * device = this->device_get(device_unique_id);
    // TODO: create 1x command per pinned segment, and callback
    device->offloader_queue_command_submit_copy<size_t, uintptr_t>(
        size,
        dst_device_unique_id,
        dst_device_addr,
        src_device_unique_id,
        src_device_addr,
        callback
    );
}

////////////////////////////
// task spawning routines //
////////////////////////////

typedef struct  memory_copy_async_args_t
{
    size_t               size;
    device_unique_id_t   dst_device_unique_id;
    uintptr_t            dst_device_addr;
    device_unique_id_t   src_device_unique_id;
    uintptr_t            src_device_addr;
}               memory_copy_async_args_t;

/* the task completes once all segment completed */
constexpr task_flag_bitfield_t  flags       = TASK_FLAG_DEVICE | TASK_FLAG_ACCESSES | TASK_FLAG_DETACHABLE;
constexpr task_access_counter_t AC          = 1;
constexpr size_t                args_size   = sizeof(memory_copy_async_args_t);

static void
body_memory_copy_async_callback(void * vargs [XKRT_CALLBACK_ARGS_MAX])
{
    runtime_t * runtime = (runtime_t *) vargs[0];
    assert(runtime);

    task_t * task = (task_t *) vargs[1];
    assert(task);

    /* retrieve task args */
    runtime->task_detachable_decr(task);
}

static void
body_memory_copy_async(runtime_t * runtime, device_t * device, task_t * task)
{
    callback_t callback;
    callback.func    = body_memory_copy_async_callback;
    callback.args[0] = runtime;
    callback.args[1] = task;

    memory_copy_async_args_t * args = (memory_copy_async_args_t *) TASK_ARGS(task);
    assert(args);

    runtime->copy(device->unique_id, args->size, args->dst_device_unique_id, args->dst_device_addr, args->src_device_unique_id, args->src_device_addr, callback);
}

void
runtime_t::memory_copy_async(
    const device_unique_id_t   device_unique_id,
    const size_t               size,
    const device_unique_id_t   dst_device_unique_id,
    const uintptr_t            dst_device_addr,
    const device_unique_id_t   src_device_unique_id,
    const uintptr_t            src_device_addr,
    int n
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    const task_format_id_t fmtid = this->formats.memory_copy_async;

    this->foreach(src_device_addr, size, n, [&] (const int i, const uintptr_t a, const uintptr_t b)
    {
        (void) i;
        assert(i < n);
        assert(a < b);
        assert(src_device_addr <= a);
        assert(b <= src_device_addr + size);

        task_t * task = this->task_new(fmtid, flags, NULL, args_size, AC);

        const size_t offset = a - src_device_addr;
        const size_t size   = b - a;

        // copy arguments
        memory_copy_async_args_t * args = (memory_copy_async_args_t *) TASK_ARGS(task);
        args->size                  = size;
        args->dst_device_unique_id  = dst_device_unique_id;
        args->dst_device_addr       = dst_device_addr + offset;
        args->src_device_unique_id  = src_device_unique_id;
        args->src_device_addr       = src_device_addr + offset;

        task_acs_info_t * acs = TASK_ACS_INFO(task);
        new (acs) task_acs_info_t(AC);

        task_det_info_t * det = TASK_DET_INFO(task);
        new (det) task_det_info_t(AC);

        task_dev_info_t * dev = TASK_DEV_INFO(task);
        new (dev) task_dev_info_t(device_unique_id, XKRT_UNSPECIFIED_TASK_ACCESS);

        # if XKRT_SUPPORT_DEBUG
        snprintf(task->label, sizeof(task->label), "memory_copy_async");
        # endif

        // detached virtual write onto the memory segment
        access_t * accesses = TASK_ACCESSES(task, flags);
        constexpr access_mode_t mode = (access_mode_t) (ACCESS_MODE_W | ACCESS_MODE_V);
        new (accesses + 0) access_t(task, a, b, mode);
        this->task_accesses_resolve(accesses, AC);

        // commit
        this->task_commit(task);
    });
}

void
memory_copy_async_register_format(runtime_t * runtime)
{
    {
        task_format_t format;
        memset(format.f, 0, sizeof(format.f));
        format.suggest = NULL;
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_memory_copy_async;
        snprintf(format.label, sizeof(format.label), "memory_copy_async");
        runtime->formats.memory_copy_async = runtime->task_format_create(&format);
    }
}

XKRT_NAMESPACE_END
