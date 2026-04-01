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

#ifndef __DRIVER_CL_H__
# define __DRIVER_CL_H__

# include <xkrt/driver/queue.h>
# include <CL/cl.h>

XKRT_NAMESPACE_BEGIN

// cl memory buffers, as opencl do not allow direct pointer arithmetic on
// device buffers and xkrt requires it
typedef struct  device_cl_buffer_t
{
    uintptr_t addr;
    size_t size;
    struct {
        cl_mem mem;
    } cl;
}               device_cl_buffer_t;

# define XKRT_DRIVER_CL_MAX_BUFFERS 8

// devices
typedef struct  device_cl_t
{
    device_t inherited;

    struct {
        cl_device_id id;
        cl_context context;
    } cl;

    struct {
        device_cl_buffer_t buffers[XKRT_DRIVER_CL_MAX_BUFFERS];
        int nbuffers;
        uintptr_t head;
    } memory;
}               device_cl_t;

typedef struct  queue_cl_t
{
    command_queue_t super;

    struct {
        cl_command_queue queue;
        cl_event * events;
    } cl;

    device_cl_t * device;

}               queue_cl_t;

typedef struct  driver_cl_t
{
    driver_t super;
}               driver_cl_t;

/* cl works on 'cl_mem' buffers and do not support pointer arithmetic directly
 * on these, so this routine returns the buffer and the offset for the given
 * addr */
void driver_cl_get_buffer_and_offset_1D(
    device_cl_t * device,
    uintptr_t addr,
    cl_mem * mem,
    size_t * offset
);

void driver_cl_get_buffer_and_offset_1D(
    device_cl_t * device,
    uintptr_t addr,
    cl_mem * mem,
    size_t * offset
);

void driver_cl_get_buffer_and_offset_2D(
    device_cl_t * device,
    uintptr_t addr,
    size_t pitch,
    cl_mem * mem,
    size_t * offset
);

XKRT_NAMESPACE_END

#endif /* __DRIVER_CL_H__ */
