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

#ifndef __DRIVER_CU_H__
# define __DRIVER_CU_H__

# include <xkrt/driver/driver.h>
# include <xkrt/driver/queue.h>
# include <cuda.h>
# include <cublas_v2.h>
# include <cusparse.h>
# include <cusolverDn.h>

XKRT_NAMESPACE_BEGIN

typedef struct  queue_cu_t
{
    command_queue_t super;

    struct {

        struct {
            CUstream  high;
            CUstream  low;
        } handle;

        struct {
            CUevent * buffer;
            xkrt_command_queue_list_counter_t capacity;
        } events;

        struct {
            cublasHandle_t handle;
        } blas;

        struct {
            cusparseHandle_t handle;
        } sparse;

        struct {
            cusolverDnHandle_t handle;
        } solver;

    } cu;
}               queue_cu_t;


typedef struct  device_cu_t
{
    device_t inherited;

    struct  {

        CUcontext context;
        CUdevice device;

        struct {
            int pciBusID;
            int pciDeviceID;
            size_t mem_total;
            char name[64];      /* GPU name */
        } prop;

    } cu;
}               device_cu_t;

typedef struct  driver_cu_t
{
    driver_t super;
}               driver_cu_t;

/* Opaque state kept behind command_batch_t::driver_handle for CUDA.
 * Forward-declared here so it can be used in command_queue_launch. */
typedef struct  command_batch_cu_handle_t
{
    CUgraph     graph;
    CUgraphExec graph_exec;

}               command_batch_cu_handle_t;

XKRT_NAMESPACE_END

#endif /* __DRIVER_CU_H__ */
