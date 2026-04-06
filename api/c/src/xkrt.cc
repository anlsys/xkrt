/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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
#include <xkrt/runtime.h>
#include <xkrt/xkrt.h>
#include <assert.h>
#include <stdlib.h>

XKRT_NAMESPACE_USE;

#ifdef __cplusplus
extern "C" {
#endif

int
xkrt_init(xkrt_runtime_t ** runtime)
{
    assert(runtime);
    runtime_t * rt = (runtime_t *) malloc(sizeof(runtime_t));
    assert(rt);
    *runtime = (xkrt_runtime_t *) rt;
    return rt->init();
}

int
xkrt_deinit(xkrt_runtime_t * runtime)
{
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    int ret = rt->deinit();
    free(rt);
    return ret;
}

void
xkrt_reset(xkrt_runtime_t * runtime)
{
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->reset();
}

int
xkrt_memory_register(
    xkrt_runtime_t * runtime,
    void * ptr,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->memory_register(ptr, size);
}

int
xkrt_memory_unregister(
    xkrt_runtime_t * runtime,
    void * ptr,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->memory_unregister(ptr, size);
}

int
xkrt_memory_register_async(
    xkrt_runtime_t * runtime,
    void * ptr,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->memory_register_async(ptr, size);
}

int
xkrt_memory_unregister_async(
    xkrt_runtime_t * runtime,
    void * ptr,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->memory_unregister_async(ptr, size);
}

int
xkrt_file_read_async(
    xkrt_runtime_t * runtime,
    int fd,
    void * buffer,
    size_t n,
    unsigned int nchunks
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->file_read_async(fd, buffer, n, nchunks);
}

int
xkrt_file_write_async(
    xkrt_runtime_t * runtime,
    int fd,
    void * buffer,
    size_t n,
    unsigned int nchunks
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->file_write_async(fd, buffer, n, nchunks);
}

void *
xkrt_memory_device_allocate(
    xkrt_runtime_t * runtime,
    device_unique_id_t device,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return (void *)rt->memory_device_allocate(device, size);
}

void
xkrt_memory_device_deallocate(
    xkrt_runtime_t * runtime,
    device_unique_id_t device,
    void * chunk
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->memory_device_deallocate(device, (area_chunk_t*)chunk);
}

void *
xkrt_memory_host_allocate(
    xkrt_runtime_t * runtime,
    device_unique_id_t device,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->memory_host_allocate(device, size);
}

void
xkrt_memory_host_deallocate(
    xkrt_runtime_t * runtime,
    device_unique_id_t device,
    void * ptr,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->memory_host_deallocate(device, ptr, size);
}

void *
xkrt_memory_unified_allocate(
    xkrt_runtime_t * runtime,
    device_unique_id_t device,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->memory_unified_allocate(device, size);
}

void
xkrt_memory_unified_deallocate(
    xkrt_runtime_t * runtime,
    device_unique_id_t device,
    void * ptr,
    size_t size
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->memory_unified_deallocate(device, ptr, size);
}

xkrt_driver_t *
xkrt_driver_get(
    xkrt_runtime_t * runtime,
    xkrt_driver_type_t type
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return (xkrt_driver_t *) rt->driver_get(type);
}

xkrt_device_t *
xkrt_device_get(
    xkrt_runtime_t * runtime,
    device_unique_id_t device
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return (xkrt_device_t *) rt->device_get(device);
}

xkrt_driver_t *
xkrt_device_driver_get(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device
) {
    assert(runtime);
    assert(device);
    device_t * dev = (device_t *) device;
    return xkrt_driver_get(runtime, dev->driver_type);
}

xkrt_device_driver_id_t
xkrt_device_driver_id_get(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device
) {
    assert(runtime);
    assert(device);
    device_t * dev = (device_t *) device;
    return dev->driver_id;
}

unsigned int
xkrt_get_ndevices(
    xkrt_runtime_t * runtime
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->get_ndevices();
}

unsigned int
xkrt_get_ndevices_max(
    xkrt_runtime_t * runtime
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->get_ndevices_max();
}

// ---------------------------
// TASKING
// ---------------------------

void
xkrt_task_commit(
    xkrt_runtime_t * runtime,
    xkrt_task_t * task
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->task_commit((task_t *) task);
}

void
xkrt_task_complete(
    xkrt_runtime_t * runtime,
    xkrt_task_t * task
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->task_complete((task_t *) task);
}

void
xkrt_task_detachable_incr(
    xkrt_runtime_t * runtime,
    xkrt_task_t * task
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->task_detachable_incr((task_t *) task);
}

void
xkrt_task_detachable_decr(
    xkrt_runtime_t * runtime,
    xkrt_task_t * task
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->task_detachable_decr((task_t *) task);
}

void
xkrt_task_enqueue(
    xkrt_runtime_t * runtime,
    xkrt_task_t * task
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    rt->task_enqueue((task_t *) task);
}

void
xkrt_task_wait(xkrt_runtime_t * runtime)
{
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return rt->task_wait();
}

void *
xkrt_task_args(xkrt_task_t * task)
{
    return TASK_ARGS((task_t *) task);
}

xkrt_access_t *
xkrt_task_accesses(xkrt_task_t * task)
{
    return (xkrt_access_t *) TASK_ACCESSES((const task_t *) task);
}

xkrt_access_t *
xkrt_task_access(
    xkrt_task_t * task,
    xkrt_task_access_counter_t access_id
) {
    access_t * accesses = TASK_ACCESSES((const task_t *) task);
    return (xkrt_access_t *) (accesses + access_id);
}

void *
xkrt_task_access_replica(
    xkrt_task_t * task,
    xkrt_task_access_counter_t access_id
) {
    access_t * accesses = TASK_ACCESSES((const task_t *) task);
    access_t * access   = accesses + access_id;
    return (void *) (access->device_view.addr);
}

xkrt_task_t *
xkrt_task_current(xkrt_runtime_t * runtime)
{
    thread_t * tls = thread_t::get_tls();
    assert(tls);
    return (xkrt_task_t *) tls->current_task;
}

// ---------------------------
// TEAM UTILITIES
// ---------------------------

xkrt_team_t *
xkrt_team_driver_device_get(
    xkrt_runtime_t * runtime,
    xkrt_driver_type_t type,
    xkrt_device_driver_id_t device_driver_id
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return (xkrt_team_t *) rt->team_get(type, device_driver_id);
}

xkrt_team_t *
xkrt_team_driver_get(
    xkrt_runtime_t * runtime,
    xkrt_driver_type_t type
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return (xkrt_team_t *) rt->team_get(type);
}

xkrt_team_t *
xkrt_team_driver_get_any(
    xkrt_runtime_t * runtime,
    xkrt_driver_type_bitfield_t types
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    return (xkrt_team_t *) rt->team_get_any(types);
}

// ---------------------------
// TEAM TASK SPAWN (C CALLBACKS)
// ---------------------------

// convert C access to C++ access
static inline void
__xkrt_set_accesses(
    task_t * task,
    const xkrt_access_t * accesses_c,
    const int naccesses,
    access_t * accesses
) {
    for (int i = 0 ; i < naccesses ; ++i)
    {
        const xkrt_access_t * access_c = accesses_c + i;
        access_t * access = accesses + i;

        switch (access_c->type)
        {
            case (ACCESS_TYPE_HANDLE):
            {
                new (access) access_t(
                    task,
                    access_c->region.handle.addr,
                    access_c->mode,
                    access_c->concurrency,
                    access_c->scope
                );
                break ;
            }

            case (ACCESS_TYPE_SEGMENT):
            {
                new (access) access_t(
                    task,
                    (const uintptr_t) access_c->region.segment.a,
                    (const uintptr_t) access_c->region.segment.b,
                    access_c->mode,
                    access_c->concurrency,
                    access_c->scope
                );
                break ;
            }

            case (ACCESS_TYPE_BLAS_MATRIX):
            {
                new (access) access_t(
                    task,
                    access_c->region.matrix.storage,
                    access_c->region.matrix.addr,
                    access_c->region.matrix.ld,
                    access_c->region.matrix.offset_m,
                    access_c->region.matrix.offset_n,
                    access_c->region.matrix.m,
                    access_c->region.matrix.n,
                    access_c->region.matrix.sizeof_type,
                    access_c->mode,
                    access_c->concurrency,
                    access_c->scope
                );
                break ;
            }

            default:
                break ;
        }
    }
}

void xkrt_task_spawn_generic(
    xkrt_runtime_t * runtime,
    const xkrt_device_unique_id_t device_unique_id,
    const xkrt_task_flag_bitfield_t flags,
    const xkrt_task_format_id_t fmtid,
    const void * args,
    const size_t args_size,
    const xkrt_access_t * accesses_c,
    const xkrt_task_access_counter_t naccesses,
    const xkrt_task_access_counter_t ocr_access,
    const xkrt_task_wait_counter_type_t   detachable_counter_initial
) {
    assert((flags & TASK_FLAG_DEVICE) || (device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID));

    // retrieve tls
    thread_t * tls = thread_t::get_tls();
    assert(tls);

    // create the task
    runtime_t * rt = (runtime_t *) runtime;
    task_t * task = rt->task_new(fmtid, flags, args, args_size, naccesses);

    if (flags & TASK_FLAG_ACCESSES)
    {
        task_acs_info_t * acs = TASK_ACS_INFO(task);
        new (acs) task_acs_info_t(naccesses);
    }

    if (flags & TASK_FLAG_DETACHABLE)
    {
        task_det_info_t * det = TASK_DET_INFO(task);
        new (det) task_det_info_t(detachable_counter_initial);
    }

    if (flags & TASK_FLAG_DEVICE)
    {
        task_dev_info_t * dev = TASK_DEV_INFO(task);
        new (dev) task_dev_info_t(device_unique_id, ocr_access);
    }

    access_t * accesses = TASK_ACCESSES(task, flags);
    __xkrt_set_accesses(task, accesses_c, naccesses, accesses);
    rt->task_accesses_resolve(accesses, naccesses);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "c-task-generic");
    # endif

    device_t * device = rt->device_get(device_unique_id);
    assert(device);

    rt->team_task_commit(device->team, task);
}

void
xkrt_task_spawn(
    xkrt_runtime_t * runtime,
    xkrt_task_func_t func,
    void * user_data
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;

    constexpr task_flag_bitfield_t flags = TASK_FLAG_ZERO;
    return rt->task_spawn<flags>(
        XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID,
        0,
        nullptr,
        nullptr,
        [func, user_data](runtime_t * runtime, device_t * device, task_t * task) {
            func((xkrt_runtime_t *) runtime, (xkrt_device_t *) device, (xkrt_task_t *) task, user_data);
        }
    );
}

void
xkrt_task_spawn_with_format_with_accesses(
    xkrt_runtime_t * runtime,
    const xkrt_device_unique_id_t device_unique_id,
    const xkrt_task_format_id_t fmtid,
    const void * args,
    const size_t args_size,
    const xkrt_access_t * accesses_c,
    const int naccesses
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;

    device_t * device = rt->device_get(device_unique_id);
    assert(device);

    team_t * team = rt->team_get(device->driver_type, device->driver_id);
    assert(team);

    assert((accesses_c == NULL && naccesses == 0) || (accesses_c && naccesses > 0));
    if (naccesses == 0)
    {
        LOGGER_FATAL("TODO");
    }
    else
    {
        constexpr task_flag_bitfield_t  flags = TASK_FLAG_ACCESSES;
        constexpr task_access_counter_t ac    = 0;
        task_t * task = rt->task_instanciate<flags>(
            XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID,
            ac,
            [&](task_t * task, access_t * accesses) { __xkrt_set_accesses(task, accesses_c, naccesses, accesses); },
            nullptr,
            args,
            args_size,
            fmtid
        );
        rt->team_task_commit(team, task);
    }
}

void
xkrt_task_spawn_with_format(
    xkrt_runtime_t * runtime,
    const xkrt_device_unique_id_t device_unique_id,
    const xkrt_task_format_id_t fmtid,
    const void * args,
    const size_t args_size
) {
    return xkrt_task_spawn_with_format_with_accesses(runtime, device_unique_id, fmtid, args, args_size, NULL, 0);
}

void
xkrt_team_task_spawn(
    xkrt_runtime_t * runtime,
    xkrt_team_t * team,
    xkrt_task_func_t func,
    void * user_data
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;

    constexpr task_flag_bitfield_t flags = TASK_FLAG_ZERO;
    rt->team_task_spawn<flags>(
        (team_t *) team,
        XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID,
        0,
        nullptr,
        nullptr,
        [func, user_data](runtime_t * runtime, device_t * device, task_t * task) {
            func((xkrt_runtime_t *) runtime, (xkrt_device_t *) device, (xkrt_task_t *) task, user_data);
        }
    );
}

void
xkrt_team_task_spawn_with_accesses(
    xkrt_runtime_t * runtime,
    xkrt_team_t * team,
    xkrt_task_func_t func,
    void * user_data,
    const xkrt_access_t * accesses_c,
    const int naccesses
) {
    assert(runtime);
    assert(accesses_c);
    assert(naccesses);
    assert(naccesses < XKRT_TASK_MAX_ACCESSES);

    runtime_t * rt = (runtime_t *) runtime;

    // spawn the task
    constexpr task_flag_bitfield_t flags = TASK_FLAG_ACCESSES;
    rt->team_task_spawn<flags>(
        (team_t *) team,
        XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID,
        (task_access_counter_t) naccesses,
        [&](task_t * task, access_t * accesses) {
            __xkrt_set_accesses(task, accesses_c, naccesses, accesses);
        },
        nullptr,
        [func, user_data](runtime_t * runtime, device_t * device, task_t * task) {
            func((xkrt_runtime_t *) runtime, (xkrt_device_t *) device, (xkrt_task_t *) task, user_data);
        }
    );
}

//-----------------
// Driver routines
//-----------------

// DRIVER META DATA

const char *
xkrt_driver_get_name(
    xkrt_driver_t * driver
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_get_name();
}

unsigned int
xkrt_driver_get_ndevices_max(
    xkrt_driver_t * driver
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_get_ndevices_max();
}

// DRIVER LIFECYCLE

int
xkrt_driver_init(
    xkrt_driver_t * driver,
    unsigned int ndevices,
    int use_p2p
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_init(ndevices, (bool) use_p2p);
}

void
xkrt_driver_finalize(
    xkrt_driver_t * driver
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_finalize();
}

// DEVICES MANAGEMENT

xkrt_device_t *
xkrt_driver_device_create(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return (xkrt_device_t *) drv->f_device_create(drv, device_driver_id);
}

void
xkrt_driver_device_init(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_device_init(device_driver_id);
}

int
xkrt_driver_device_commit(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_device_unique_id_bitfield_t * affinity
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_device_commit(device_driver_id, affinity);
}

int
xkrt_driver_device_destroy(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_device_destroy(device_driver_id);
}

void
xkrt_driver_device_info(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    char * buffer,
    size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_device_info(device_driver_id, buffer, size);
}

// MEMORY MANAGEMENT

# if 0
void
xkrt_driver_memory_device_info(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX],
    int * nmemories
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_memory_device_info(device_driver_id, info, nmemories);
}
# endif

void *
xkrt_driver_memory_device_allocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    const size_t size,
    int area_idx
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_device_allocate(device_driver_id, size, area_idx);
}

void
xkrt_driver_memory_device_deallocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    void * ptr,
    const size_t size,
    int area_idx
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_memory_device_deallocate(device_driver_id, ptr, size, area_idx);
}

void *
xkrt_driver_memory_host_allocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_host_allocate(device_driver_id, size);
}

void
xkrt_driver_memory_host_deallocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    void * mem,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_memory_host_deallocate(device_driver_id, mem, size);
}

void *
xkrt_driver_memory_unified_allocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_unified_allocate(device_driver_id, size);
}

void
xkrt_driver_memory_unified_deallocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    void * mem,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_memory_unified_deallocate(device_driver_id, mem, size);
}

int
xkrt_driver_memory_host_register(
    xkrt_driver_t * driver,
    void * mem,
    uint64_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_host_register(mem, size);
}

int
xkrt_driver_memory_host_unregister(
    xkrt_driver_t * driver,
    void * mem,
    uint64_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_host_unregister(mem, size);
}

int
xkrt_driver_memory_unified_advise_device(
    xkrt_driver_t * driver,
    const xkrt_device_driver_id_t device_unique_id,
    const void * addr,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_unified_advise_device(device_unique_id, addr, size);
}

int
xkrt_driver_memory_unified_advise_host(
    xkrt_driver_t * driver,
    const void * addr,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_unified_advise_host(addr, size);
}

int
xkrt_driver_memory_unified_prefetch_device(
    xkrt_driver_t * driver,
    const xkrt_device_driver_id_t device_unique_id,
    const void * addr,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_unified_prefetch_device(device_unique_id, addr, size);
}

int
xkrt_driver_memory_unified_prefetch_host(
    xkrt_driver_t * driver,
    const void * addr,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_memory_unified_prefetch_host(addr, size);
}

// MEMORY TRANSFERS

int
xkrt_driver_transfer_h2d(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_transfer_h2d(dst, src, size);
}

int
xkrt_driver_transfer_d2h(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_transfer_d2h(dst, src, size);
}

int
xkrt_driver_transfer_d2d(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_transfer_d2d(dst, src, size);
}

int
xkrt_driver_transfer_h2d_async(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size,
    xkrt_command_queue_t * queue
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_transfer_h2d_async(dst, src, size, (command_queue_t *) queue);
}

int
xkrt_driver_transfer_d2h_async(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size,
    xkrt_command_queue_t * queue
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_transfer_d2h_async(dst, src, size, (command_queue_t *) queue);
}

int
xkrt_driver_transfer_d2d_async(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size,
    xkrt_command_queue_t * queue
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_transfer_d2d_async(dst, src, size, (command_queue_t *) queue);
}

// KERNEL LAUNCH

void
xkrt_task_prog_launch(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device,
    xkrt_task_t * task,
    int synchronous,
    xkrt_prog_launcher_t launcher
) {
    assert(runtime);
    runtime_t * rt = (runtime_t *) runtime;
    task_t * t = (task_t *) task;
    device_t * dev = (device_t *) device;

    constexpr command_queue_type_t qtype = XKRT_QUEUE_TYPE_KERN;
    constexpr ocg::command_type_t  ctype = ocg::COMMAND_TYPE_PROG;
    const     command_flag_t       flags = synchronous ? COMMAND_FLAG_SYNCHRONOUS : COMMAND_FLAG_NONE;
    rt->task_emit_command(
        dev->unique_id,
        qtype,
        ctype,
        flags,
        [&] (xkrt::command_t * cmd) {
            /* create a new kernel command */
            cmd->prog.launcher.fixed.fn = (void (*)(void * [OCG_CALLBACK_ARGS_MAX])) launcher;
            cmd->prog.launcher.fixed.args[0] = rt;
            cmd->prog.launcher.fixed.args[1] = task;
            static_assert(OCG_CALLBACK_ARGS_MAX >= 2);

            /* program launcher */
            cmd->flags = cmd->flags | COMMAND_FLAG_PROG_LAUNCHER;
        }
    );
}

// THREADING

# if 0
int
xkrt_driver_device_cpuset(
    xkrt_driver_t * driver,
    hwloc_topology_t topology,
    cpu_set_t * cpuset,
    xkrt_device_driver_id_t device_driver_id
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_device_cpuset(topology, cpuset, device_driver_id);
}
# endif

// QUEUE MANAGEMENT

int
xkrt_driver_command_queue_suggest(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_command_queue_type_t qtype
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return drv->f_command_queue_suggest(device_driver_id, qtype);
}

xkrt_command_queue_t *
xkrt_driver_command_queue_create(
    xkrt_driver_t * driver,
    xkrt_device_t * device,
    xkrt_command_queue_type_t qtype,
    xkrt_command_queue_list_counter_t capacity
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return (xkrt_command_queue_t *) drv->f_command_queue_create((device_t *) device, qtype, capacity);
}

void
xkrt_driver_command_queue_delete(
    xkrt_driver_t * driver,
    xkrt_command_queue_t * queue
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_command_queue_delete((command_queue_t *) queue);
}

// MODULES

xkrt_driver_module_t
xkrt_driver_module_load(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    uint8_t * bin,
    size_t binsize,
    xkrt_driver_module_format_t format
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return (xkrt_driver_module_t) drv->f_module_load(device_driver_id, bin, binsize, format);
}

void
xkrt_driver_module_unload(
    xkrt_driver_t * driver,
    xkrt_driver_module_t module
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_module_unload(module);
}

xkrt_driver_module_fn_t
xkrt_driver_module_get_fn(
    xkrt_driver_t * driver,
    xkrt_driver_module_t module,
    const char * name
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    return (xkrt_driver_module_fn_t) drv->f_module_get_fn(module, name);
}

// ENERGY COUNTER

void
xkrt_driver_power_start(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_power_t * pwr
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_power_start(device_driver_id, pwr);
}

void
xkrt_driver_power_stop(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_power_t * pwr
) {
    assert(driver);
    driver_t * drv = (driver_t *) driver;
    drv->f_power_stop(device_driver_id, pwr);
}

// ---------------------------
// TASK FORMATS
// ---------------------------

xkrt_task_format_id_t
xkrt_task_format_put(
    xkrt_runtime_t * runtime,
    const char * label
) {
    runtime_t * rt = (runtime_t *) runtime;
    return rt->task_format_put(label);
}

int
xkrt_task_format_set(
    xkrt_runtime_t * runtime,
    xkrt_task_format_id_t fmtid,
    xkrt_task_format_target_t target,
    xkrt_task_format_func_t func
) {
    runtime_t * rt = (runtime_t *) runtime;
    return rt->task_format_set(fmtid, target, func);
}

// ---------------------
// LOGGER
// ---------------------

void
xkrt_logger_info(const char * msg)
{
    LOGGER_INFO("%s", msg);
}

void
xkrt_logger_debug(const char * msg)
{
    LOGGER_DEBUG("%s", msg);
}

void
xkrt_logger_warn(const char * msg)
{
    LOGGER_WARN("%s", msg);
}

void
xkrt_logger_error(const char * msg)
{
    LOGGER_ERROR("%s", msg);
}

void
xkrt_logger_fatal(const char * msg)
{
    LOGGER_FATAL("%s", msg);
}

#ifdef __cplusplus
} // extern "C"
#endif

