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

#ifndef __XKRT_INTERNALS_H__
# define __XKRT_INTERNALS_H__

# include <xkrt/runtime.h>

XKRT_NAMESPACE_BEGIN

///////////////
// Utilities //
///////////////

/**
 * @brief Register the host capture task format with the runtime
 * @param runtime Pointer to the runtime system
 */
void task_host_capture_register_format(runtime_t * runtime);

/**
 * @brief Register the asynchronous memory copy format
 * @param runtime Pointer to the runtime system
 */
void memory_copy_async_register_format(runtime_t * runtime);

/**
 * @brief Register the v2 asynchronous memory registration format
 * @param runtime Pointer to the runtime system
 */
void memory_register_async_register_format(runtime_t * runtime);

/**
 * @brief Register the asynchronous file operation format
 * @param runtime Pointer to the runtime system
 */
void file_async_register_format(runtime_t * runtime);

/**
 * @brief Main routine for device team threads
 * @param runtime Pointer to the runtime system
 * @param team Pointer to the device team
 * @param thread Pointer to the thread structure
 * @return Thread return value (typically nullptr)
 */
void * device_thread_main(runtime_t * runtime, team_t * team, thread_t * thread);

/**
 * @brief Initialize all device drivers
 * @param runtime Pointer to the runtime system
 */
void drivers_init(runtime_t * runtime);

/**
 * @brief Deinitialize and cleanup all device drivers
 * @param runtime Pointer to the runtime system
 */
void drivers_deinit(runtime_t * runtime);

/**
 * @brief Execute a task after all data accesses have been resolved
 * @param runtime Pointer to the runtime system
 * @param device Pointer to the device executing the task
 * @param task Pointer to the task to execute
 */
void task_execute(
    runtime_t * runtime,
    device_t * device,
    task_t * task
);

/**
 * @brief Arguments passed to device team threads during creation
 */
typedef struct  device_team_args_t
{
    driver_t * driver;                      ///< Pointer to the device driver
    device_unique_id_t device_unique_id;    ///< Global device identifier
    device_driver_id_t device_driver_id;    ///< Driver-specific device identifier
    pthread_barrier_t barrier;              ///< Synchronization barrier for team initialization
}               device_team_args_t;

/**
 * @brief Get the memory coherency controller for a task's data access
 * @param runtime Pointer to the runtime system
 * @param task Pointer to the task
 * @param access Pointer to the data access descriptor
 * @return Pointer to the memory coherency controller managing this access
 */
MemoryCoherencyController * task_get_memory_controller(
    runtime_t * runtime,
    task_t * task,
    const access_t * access
);

XKRT_NAMESPACE_END

#endif /* __XKRT_INTERNALS_H__ */
