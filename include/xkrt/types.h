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

#ifndef __XKRT_TYPES_HPP__
# define __XKRT_TYPES_HPP__

# include <atomic>
# include <functional>

# include <xkrt/consts.h>
# include <xkrt/driver/driver-module.h>
# include <xkrt/driver/kernel-launcher.h>
# include <xkrt/driver/power.h>
# include <xkrt/driver/queue-command-list-counter.h>
# include <xkrt/driver/queue-type.h>
# include <xkrt/driver/driver-type.h>
# include <xkrt/memory/size.h>
# include <xkrt/memory/access/blas/matrix-storage.h>
# include <xkrt/memory/access/concurrency.h>
# include <xkrt/memory/access/mode.h>
# include <xkrt/memory/access/scope.h>
# include <xkrt/memory/access/type.h>
# include <xkrt/namespace.h>
# include <xkrt/task/flag.h>
# include <xkrt/task/format.h>
# include <xkrt/task/state.h>

XKRT_NAMESPACE_BEGIN

typedef xkrt_task_wait_counter_type_t           task_wait_counter_type_t;
typedef std::atomic<task_wait_counter_type_t>   task_wait_counter_t;
typedef xkrt_task_access_counter_t              task_access_counter_t;
typedef xkrt_task_flags_t                       task_flags_t;
typedef xkrt_task_state_t                       task_state_t;
typedef xkrt_task_flag_bitfield_t               task_flag_bitfield_t;

typedef xkrt_device_driver_id_t                 device_driver_id_t;
typedef xkrt_device_unique_id_t                 device_unique_id_t;
typedef xkrt_device_unique_id_bitfield_t        device_unique_id_bitfield_t;

typedef xkrt_driver_type_t                      driver_type_t;
typedef xkrt_driver_type_bitfield_t             driver_type_bitfield_t;

typedef xkrt_access_concurrency_t               access_concurrency_t;
typedef xkrt_access_mode_t                      access_mode_t;
typedef xkrt_access_scope_t                     access_scope_t;
typedef xkrt_access_type_t                      access_type_t;
typedef xkrt_matrix_storage_t                   matrix_storage_t;

typedef xkrt_task_formats_t                     task_formats_t;
typedef xkrt_task_format_t                      task_format_t;
typedef xkrt_task_format_id_t                   task_format_id_t;
typedef xkrt_task_format_func_t                 task_format_func_t;
typedef xkrt_task_format_target_t               task_format_target_t;

typedef xkrt_command_callback_index_t           command_callback_index_t;
typedef xkrt_command_queue_type_t               command_queue_type_t;
typedef xkrt_command_queue_list_counter_t       command_queue_list_counter_t;

typedef xkrt_driver_module_fn_t                 driver_module_fn_t;
typedef xkrt_driver_module_format_t             driver_module_format_t;
typedef xkrt_driver_module_t                    driver_module_t;

typedef xkrt_power_counter_t                    power_counter_t;
typedef xkrt_power_t                            power_t;

typedef xkrt_prog_launcher_t                    prog_launcher_t;

typedef xkrt_memory_size_t                      memory_size_t;
typedef xkrt_memory_size_unit_t                 memory_size_unit_t;
typedef xkrt_memory_size_type_t                 memory_size_type_t;

XKRT_NAMESPACE_END

# endif /* __XKRT_TASK_HPP__ */
