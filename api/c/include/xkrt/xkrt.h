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

#ifndef __XKRT_C_H__
#define __XKRT_C_H__

#include <xkrt/consts.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

# include <xkrt/driver/driver-module.h>
# include <xkrt/driver/driver-type.h>
# include <xkrt/driver/kernel-launcher.h>
# include <xkrt/driver/power.h>
# include <xkrt/driver/queue-command-list-counter.h>
# include <xkrt/driver/queue-type.h>
# include <xkrt/memory/access/blas/matrix-storage.h>
# include <xkrt/memory/access/concurrency.h>
# include <xkrt/memory/access/mode.h>
# include <xkrt/memory/access/scope.h>
# include <xkrt/memory/access/type.h>
# include <xkrt/task/flag.h>
# include <xkrt/task/format.h>
# include <xkrt/task/state.h>

/* Types */
typedef void * xkrt_runtime_t;
typedef void * xkrt_task_t;
typedef void * xkrt_team_t;
typedef void * xkrt_driver_t;
typedef void * xkrt_device_t;
typedef void * xkrt_command_queue_t;
typedef void * command_t;

typedef void (*xkrt_task_func_t)(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device,
    xkrt_task_t * task,
    void * user_data
);

typedef struct  xkrt_access_t
{
    xkrt_access_concurrency_t concurrency;
    xkrt_access_mode_t        mode;
    xkrt_access_scope_t       scope;
    xkrt_access_type_t        type;
    union xkrt_region_t {
        struct xkrt_handle_t {
            void * addr;
        } handle;
        struct xkrt_segment_t {
            void * a;
            void * b;
        } segment;
        struct xkrt_matrix_t {
            xkrt_matrix_storage_t storage;
            void * addr;
            unsigned long ld;
            unsigned long offset_m;
            unsigned long offset_n;
            unsigned long m;
            unsigned long n;
            unsigned long sizeof_type;
        } matrix;
    } region;
}               xkrt_access_t;

/* Runtime init */
int  xkrt_init  (xkrt_runtime_t ** runtime);
int  xkrt_deinit(xkrt_runtime_t  * runtime);
void xkrt_reset (xkrt_runtime_t  * runtime);

/* Memory registration */
int  xkrt_memory_register        (xkrt_runtime_t * runtime, void * ptr, size_t size);
int  xkrt_memory_unregister      (xkrt_runtime_t * runtime, void * ptr, size_t size);
int  xkrt_memory_register_async  (xkrt_runtime_t * runtime, void * ptr, size_t size);
int  xkrt_memory_unregister_async(xkrt_runtime_t * runtime, void * ptr, size_t size);

/* File operations */
int  xkrt_file_read_async (xkrt_runtime_t * runtime, int fd, void * buffer, size_t n, unsigned int nchunks);
int  xkrt_file_write_async(xkrt_runtime_t * runtime, int fd, void * buffer, size_t n, unsigned int nchunks);

/* Device memory allocation */
void * xkrt_memory_device_allocate   (xkrt_runtime_t * runtime, xkrt_device_unique_id_t device, size_t size);
void   xkrt_memory_device_deallocate (xkrt_runtime_t * runtime, xkrt_device_unique_id_t device, void * chunk);
void * xkrt_memory_host_allocate     (xkrt_runtime_t * runtime, xkrt_device_unique_id_t device, size_t size);
void   xkrt_memory_host_deallocate   (xkrt_runtime_t * runtime, xkrt_device_unique_id_t device, void * ptr, size_t size);
void * xkrt_memory_unified_allocate  (xkrt_runtime_t * runtime, xkrt_device_unique_id_t device, size_t size);
void   xkrt_memory_unified_deallocate(xkrt_runtime_t * runtime, xkrt_device_unique_id_t device, void * ptr, size_t size);

/* Drivers/devices */
xkrt_driver_t * xkrt_driver_get(xkrt_runtime_t * runtime, xkrt_driver_type_t type);
xkrt_device_t * xkrt_device_get(xkrt_runtime_t * runtime, xkrt_device_unique_id_t device);

xkrt_driver_t * xkrt_device_driver_get(xkrt_runtime_t * runtime, xkrt_device_t * device);
xkrt_device_driver_id_t xkrt_device_driver_id_get(xkrt_runtime_t * runtime, xkrt_device_t * device);

unsigned int xkrt_get_ndevices    (xkrt_runtime_t * runtime);
unsigned int xkrt_get_ndevices_max(xkrt_runtime_t * runtime);

/* TASKING */
void xkrt_task_commit  (xkrt_runtime_t * runtime, xkrt_task_t * task);
void xkrt_task_complete(xkrt_runtime_t * runtime, xkrt_task_t * task);

void xkrt_task_detachable_incr(xkrt_runtime_t * runtime, xkrt_task_t * task);
void xkrt_task_detachable_decr(xkrt_runtime_t * runtime, xkrt_task_t * task);

void xkrt_task_enqueue(xkrt_runtime_t * runtime, xkrt_task_t * task);

void xkrt_task_wait(xkrt_runtime_t * runtime);

/* utilities to retrieve task properties */
void          * xkrt_task_args(xkrt_task_t * task);
xkrt_access_t * xkrt_task_accesses(xkrt_task_t * task);

/* task accesses utilities */
xkrt_access_t * xkrt_task_access(xkrt_task_t * task, xkrt_task_access_counter_t access_id);
void          * xkrt_task_access_replica(xkrt_task_t * task, xkrt_task_access_counter_t access_id);

xkrt_task_t * xkrt_task_current(xkrt_runtime_t * runtime);

/* TASK SPAWN */

void xkrt_task_spawn_generic(
    xkrt_runtime_t * runtime,
    const xkrt_device_unique_id_t device_unique_id,
    const xkrt_task_flag_bitfield_t flags,
    const xkrt_task_format_id_t fmtid,
    const void * args,
    const size_t args_size,
    const xkrt_access_t * accesses,
    const xkrt_task_access_counter_t naccesses,
    const xkrt_task_access_counter_t ocr_access,
    const xkrt_task_wait_counter_type_t detachable_counter_initial
);

void xkrt_task_spawn_with_format(
    xkrt_runtime_t * runtime,
    const xkrt_device_unique_id_t device_unique_id,
    const xkrt_task_format_id_t fmtid,
    const void * args,
    const size_t args_size
);

void xkrt_task_spawn_with_format_with_accesses(
    xkrt_runtime_t * runtime,
    const xkrt_device_unique_id_t device_unique_id,
    const xkrt_task_format_id_t fmtid,
    const void * args,
    const size_t args_size,
    const xkrt_access_t * accesses,
    const int naccesses
);

void xkrt_task_spawn(
    xkrt_runtime_t * runtime,
    xkrt_task_func_t func,
    void * user_data
);

void xkrt_task_prog_launch(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device,
    xkrt_task_t * task,
    int synchronous,
    xkrt_prog_launcher_t launcher
);

/* TEAM UTILITIES */
xkrt_team_t * xkrt_team_driver_device_get(xkrt_runtime_t * runtime, xkrt_driver_type_t type, xkrt_device_driver_id_t device_driver_id);
xkrt_team_t * xkrt_team_driver_get       (xkrt_runtime_t * runtime, xkrt_driver_type_t type);
xkrt_team_t * xkrt_team_driver_get_any   (xkrt_runtime_t * runtime, xkrt_driver_type_bitfield_t types);

/* TEAM TASK SPAWN */
void xkrt_team_task_spawn(
    xkrt_runtime_t * runtime,
    xkrt_team_t * team,
    xkrt_task_func_t func,
    void * user_data
);

void xkrt_team_task_spawn_with_accesses(
    xkrt_runtime_t * runtime,
    xkrt_team_t * team,
    xkrt_task_func_t func,
    void * user_data,
    const xkrt_access_t * accesses,
    const int naccesses
);

/* Task format */
xkrt_task_format_id_t xkrt_task_format_put(xkrt_runtime_t * runtime, const char * label);
int xkrt_task_format_set(xkrt_runtime_t * runtime, xkrt_task_format_id_t fmtid, xkrt_task_format_target_t target, xkrt_task_format_func_t func);

// DRIVER META DATA

const char *
xkrt_driver_get_name(
    xkrt_driver_t * driver
);

unsigned int
xkrt_driver_get_ndevices_max(
    xkrt_driver_t * driver
);

// DRIVER LIFECYCLE

int
xkrt_driver_init(
    xkrt_driver_t * driver,
    unsigned int ndevices,
    int use_p2p
);

void
xkrt_driver_finalize(
    xkrt_driver_t * driver
);

// DEVICES MANAGEMENT

xkrt_device_t *
xkrt_driver_device_create(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id
);

void
xkrt_driver_device_init(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id
);

int
xkrt_driver_device_commit(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_device_unique_id_bitfield_t * affinity
);

int
xkrt_driver_device_destroy(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id
);

void
xkrt_driver_device_info(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    char * buffer,
    size_t size
);

// MEMORY MANAGEMENT

# if 0
void
xkrt_driver_memory_device_info(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX],
    int * nmemories
);
# endif

void *
xkrt_driver_memory_device_allocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    const size_t size,
    int area_idx
);

void
xkrt_driver_memory_device_deallocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    void * ptr,
    const size_t size,
    int area_idx
);

void *
xkrt_driver_memory_host_allocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    const size_t size
);

void
xkrt_driver_memory_host_deallocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    void * mem,
    const size_t size
);

void *
xkrt_driver_memory_unified_allocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    const size_t size
);

void
xkrt_driver_memory_unified_deallocate(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    void * mem,
    const size_t size
);

int
xkrt_driver_memory_host_register(
    xkrt_driver_t * driver,
    void * mem,
    uint64_t size
);

int
xkrt_driver_memory_host_unregister(
    xkrt_driver_t * driver,
    void * mem,
    uint64_t size
);

int
xkrt_driver_memory_unified_advise_device(
    xkrt_driver_t * driver,
    const xkrt_device_driver_id_t device_unique_id,
    const void * addr,
    const size_t size
);

int
xkrt_driver_memory_unified_advise_host(
    xkrt_driver_t * driver,
    const void * addr,
    const size_t size
);

int
xkrt_driver_memory_unified_prefetch_device(
    xkrt_driver_t * driver,
    const xkrt_device_driver_id_t device_unique_id,
    const void * addr,
    const size_t size
);

int
xkrt_driver_memory_unified_prefetch_host(
    xkrt_driver_t * driver,
    const void * addr,
    const size_t size
);

// MEMORY TRANSFERS

int
xkrt_driver_transfer_h2d(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size
);

int
xkrt_driver_transfer_d2h(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size
);

int
xkrt_driver_transfer_d2d(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size
);

int
xkrt_driver_transfer_h2d_async(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size,
    xkrt_command_queue_t * queue
);

int
xkrt_driver_transfer_d2h_async(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size,
    xkrt_command_queue_t * queue
);

int
xkrt_driver_transfer_d2d_async(
    xkrt_driver_t * driver,
    void * dst,
    void * src,
    const size_t size,
    xkrt_command_queue_t * queue
);

// THREADING

# if 0
int
xkrt_driver_device_cpuset(
    xkrt_driver_t * driver,
    hwloc_topology_t topology,
    cpu_set_t * cpuset,
    xkrt_device_driver_id_t device_driver_id
);
# endif

// QUEUE MANAGEMENT

int
xkrt_driver_command_queue_suggest(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_command_queue_type_t qtype
);

xkrt_command_queue_t *
xkrt_driver_command_queue_create(
    xkrt_driver_t * driver,
    xkrt_device_t * device,
    xkrt_command_queue_type_t qtype,
    xkrt_command_queue_list_counter_t capacity
);

void
xkrt_driver_command_queue_delete(
    xkrt_driver_t * driver,
    xkrt_command_queue_t * queue
);

// MODULES

xkrt_driver_module_t
xkrt_driver_module_load(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    uint8_t * bin,
    size_t binsize,
    xkrt_driver_module_format_t format
);

void
xkrt_driver_module_unload(
    xkrt_driver_t * driver,
    xkrt_driver_module_t module
);

xkrt_driver_module_fn_t
xkrt_driver_module_get_fn(
    xkrt_driver_t * driver,
    xkrt_driver_module_t module,
    const char * name
);

// ENERGY COUNTER

void
xkrt_driver_power_start(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_power_t * pwr
);

void
xkrt_driver_power_stop(
    xkrt_driver_t * driver,
    xkrt_device_driver_id_t device_driver_id,
    xkrt_power_t * pwr
);

// LOGGER
void xkrt_logger_info (const char * msg);
void xkrt_logger_debug(const char * msg);
void xkrt_logger_warn (const char * msg);
void xkrt_logger_error(const char * msg);
void xkrt_logger_fatal(const char * msg);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __XKRT_C_H__ */

