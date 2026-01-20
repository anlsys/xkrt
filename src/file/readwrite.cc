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

typedef struct  file_args_t
{
    int fd;
    size_t offset;
    void * buffer;
    size_t size;
}               file_args_t;

/* the task completes once all segment completed */
constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT | TASK_FLAG_DEVICE | TASK_FLAG_DETACHABLE;
constexpr unsigned int ac  = 1;
constexpr size_t task_size = task_compute_size(flags, ac);
constexpr size_t args_size = sizeof(file_args_t);

static void
body_file_async_callback(void * vargs [XKRT_CALLBACK_ARGS_MAX])
{
    runtime_t * runtime = (runtime_t *) vargs[0];
    assert(runtime);

    task_t * task = (task_t *) vargs[1];
    assert(task);

    /* retrieve task args */
    runtime->task_detachable_decr(task);
}

template<command_type_t T>
static void
body_file_async(runtime_t * runtime, device_t * device, task_t * task)
{
    assert(runtime);
    assert(device);
    assert(task);

    callback_t callback;
    callback.func    = body_file_async_callback;
    callback.args[0] = runtime;
    callback.args[1] = task;

    file_args_t * args = (file_args_t *) TASK_ARGS(task, task_size);

    /* task completion must wait for detachable event to reach 0, i.e., want
     * the `body_file_async_callback` was called, when the file had been read */
    runtime->task_detachable_incr(task);

    /* submit a file i/o command */
    device->offloader_queue_command_submit_file<T>(
        args->fd, args->buffer, args->size, args->offset, callback
    );
}

// TODO: reimplement using partitionned dependencies
template<command_type_t T>
static inline int
file_async(
    runtime_t * runtime,
    int fd,
    void * buffer,
    const size_t size,
    int n
) {
    if (n > XKRT_IO_URING_DEPTH)
        LOGGER_WARN("Spawning n=%d tasks, while io_uring queues were configured with a XKRT_IO_URING_DEPTH=%d", n, XKRT_IO_URING_DEPTH);

    static_assert(T == COMMAND_TYPE_FD_READ || T == COMMAND_TYPE_FD_WRITE);
    assert(n > 0);

    // compute number of commands to spawn
    if (size < (size_t) n)
       n = (int) size;

    // compute chunk size
    const size_t chunksize = size / n;
    assert(chunksize > 0);

    // create the task that submit the i/o command
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    // get task format
    const task_format_id_t fmtid = (T == COMMAND_TYPE_FD_READ) ? runtime->formats.file_read_async : runtime->formats.file_write_async;

    const uintptr_t p = (const uintptr_t) buffer;

    runtime->foreach(p, size, n, [&] (const int i, const uintptr_t a, const uintptr_t b)
    {
        (void) i;

        assert(p <= a);
        assert(p <  b);
        assert(a <  b);

        task_t * task = runtime->task_new(fmtid, flags, task_size + args_size);

        // copy arguments
        file_args_t * args = (file_args_t *) TASK_ARGS(task, task_size);
        args->fd = fd;
        args->offset = a - p;
        args->buffer = (void *) a;
        args->size   = (size_t) (b - a);

        task_dep_info_t * dep = TASK_DEP_INFO(task);
        new (dep) task_dep_info_t(ac);

        task_det_info_t * det = TASK_DET_INFO(task);
        new (det) task_det_info_t(0);

        task_dev_info_t * dev = TASK_DEV_INFO(task);
        new (dev) task_dev_info_t(HOST_DEVICE_GLOBAL_ID, UNSPECIFIED_TASK_ACCESS);

        # if XKRT_SUPPORT_DEBUG
        snprintf(task->label, sizeof(task->label), T == COMMAND_TYPE_FD_READ ? "fread" : "fwrite");
        # endif

        // detached virtual write onto the memory segment
        access_t * accesses = TASK_ACCESSES(task, flags);
        constexpr access_mode_t mode = (access_mode_t) (ACCESS_MODE_W | ACCESS_MODE_V);
        new (accesses + 0) access_t(task, a, b, mode);
        thread->resolve(accesses, 1);

        // commit
        runtime->task_commit(task);
    });

    return 0;
}

int
runtime_t::file_read_async(
    int fd,
    void * buffer,
    const size_t size,
    int n
) {
    return file_async<COMMAND_TYPE_FD_READ>(this, fd, buffer, size, n);
}

int
runtime_t::file_write_async(
    int fd,
    void * buffer,
    const size_t size,
    int n
) {
    return file_async<COMMAND_TYPE_FD_WRITE>(this, fd, buffer, size, n);
}

void
file_async_register_format(runtime_t * runtime)
{
    {
        task_format_t format;
        memset(&format, 0, sizeof(format));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_file_async<COMMAND_TYPE_FD_READ>;
        snprintf(format.label, sizeof(format.label), "file_read_async");
        runtime->formats.file_read_async = runtime->task_format_create(&format);
    }

    {
        task_format_t format;
        memset(&format, 0, sizeof(format));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_file_async<COMMAND_TYPE_FD_WRITE>;
        snprintf(format.label, sizeof(format.label), "file_write_async");
        runtime->formats.file_write_async = runtime->task_format_create(&format);
    }
}

XKRT_NAMESPACE_END;
