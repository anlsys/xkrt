/*
** Copyright 2024,2025 INRIA
**
** Contributors :
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

#ifndef __DRIVER_ZE_H__
# define __DRIVER_ZE_H__

# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/queue.h>

# include <xkrt/support.h>

# include <ze_api.h>
# if XKRT_SUPPORT_ZE_SYCL_INTEROP
#  include <sycl/sycl.hpp>
# endif /* XKRT_SUPPORT_ZE_SYCL_INTEROP */
# if XKRT_SUPPORT_ZES
#  include <zes_api.h>
# endif

XKRT_NAMESPACE_BEGIN

typedef enum    device_ze_type_t
{
    DEVICE_ZE_TYPE_AURORA25,    /* max gpu series with buggy 2d copy */
    DEVICE_ZE_TYPE_UNKNOWN
}               device_ze_type_t;

typedef struct  device_ze_t
{
    device_t inherited;

    device_ze_type_t type;

    struct {

        // handles
        ze_driver_handle_t      driver;
        ze_context_handle_t     context;
        ze_device_handle_t      handle;
        ze_device_properties_t  properties;

        // indexes
        struct {
            unsigned int driver;    // ze driver index
            unsigned int device;    // ze device index
        } index;

        // number of command queue group
        uint32_t ncommandqueuegroups;

        // per command queue group property
        ze_command_queue_group_properties_t * command_queue_group_properties;

        // per command queue number of queue used
        std::atomic<uint32_t> * command_queue_group_used;

        // memory properties
        struct {
            uint32_t count;
            ze_device_memory_properties_t properties[XKRT_DEVICE_MEMORIES_MAX];
        } memory;

    } ze;

    # if XKRT_SUPPORT_ZE_SYCL_INTEROP
    // sycl interop
    struct {
        sycl::device device;
        sycl::context context;
    } sycl;
    # endif /* XKRT_SUPPORT_ZE_SYCL_INTEROP */

    # if XKRT_SUPPORT_ZES
    struct {

        // handles
        zes_device_handle_t device; // zes device

        // indexes
        struct {
            unsigned int driver;    // ze driver index
            unsigned int device;    // ze device index
            ze_bool_t on_subdevice; // if this is a subdevice
            uint32_t subdevice_id;  // subdevice id, if it is a subdevice
        } index;

        // memory
        struct {
            uint32_t count;
            zes_mem_handle_t handles[XKRT_DEVICE_MEMORIES_MAX];
        } memory;

        // pwr
        struct {
            zes_pwr_handle_t handle;
        } pwr;
    } zes;

    # endif /* XKRT_SUPPORT_ZES */

}               device_ze_t;


typedef struct  queue_ze_t
{
    command_queue_t super;

    struct {
        struct {
            ze_command_list_handle_t list;
        } command;
        struct {
            ze_event_pool_handle_t  pool;
            ze_event_handle_t     * list;
        } events;

        // bad design, but required to submit kernels with level zero
        device_ze_t * device;

    } ze;

    # if XKRT_SUPPORT_ZE_SYCL_INTEROP
    struct {
        sycl::queue queue;
    } sycl;
    # endif /* XKRT_SUPPORT_ZE_SYCL_INTEROP */

}               queue_ze_t;


typedef struct  driver_ze_t
{
    driver_t super;
}               driver_ze_t;

XKRT_NAMESPACE_END

#endif /* __DRIVER_ZE_H__ */
