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

#ifndef __OPENCG_COMMAND_HPP__
# define __OPENCG_COMMAND_HPP__

# include <opencg/opencg.hpp>
# include <opencg/device-type.hpp>
# include <opencg/namespace.hpp>

# include <stddef.h>

OCG_NAMESPACE_BEGIN

/* Move data between devices */
struct command_copy_1D_t
{
    device_unique_id_t src_device_unique_id;
    device_unique_id_t dst_device_unique_id;
    uintptr_t src_device_addr;
    uintptr_t dst_device_addr;
    size_t size;
};

struct command_copy_2D_t
{
    device_unique_id_t src_device_unique_id;
    device_unique_id_t dst_device_unique_id;
    uintptr_t src_addr;
    size_t src_ld;
    uintptr_t dst_addr;
    size_t dst_ld;
    size_t m;
    size_t n;
    size_t sizeof_type;
};

enum command_prog_source_type_t
{
    /* LLVM IR */
    COMMAND_PROG_SOURCE_TYPE_LLVMIR,

    /* MLIR */
    COMMAND_PROG_SOURCE_TYPE_MLIR,

    /* PTX */
    COMMAND_PROG_SOURCE_TYPE_PTX,

    /* CL (OpenCL prog language) */
    COMMAND_PROG_SOURCE_TYPE_CL,

    /* SPIRV */
    COMMAND_PROG_SOURCE_TYPE_SPIRV

};

/* a prog to be submitted via (cuKernelLaunch, ...) */
struct command_prog_t
{
    /* Program function and arguments */
    union {

        /* Fixed argument sizes */
        struct {
            # define OCG_CALLBACK_ARGS_MAX 3
            void (*fn)(void * [OCG_CALLBACK_ARGS_MAX]);
            void * args[OCG_CALLBACK_ARGS_MAX];
        } fixed;

        /* variadic argument sizes */
        struct {
            void * fn;
            void * args;
            size_t args_size;
        } variadic;

    } launcher;

    // TODO: shouldnt this bellow be user-defined instead ?

    /* source of the prog */
    void * source;

    /* format of the prog */
    command_prog_source_type_t source_type;

    /* grid parameters */
    struct {
        unsigned int x, y, z;
    } grid;

    /* block dimension */
    struct {
        unsigned int x, y, z;
    } block;
};

// TODO: remove prog_launcher, which is XKRT-specific, and implement it as a 'host' program in XKRT

/* to launch progs on a device with a routine opaquely pushing to a queue (cublas...) */
struct command_prog_launcher_t
{
    // arguments are:
    //   runtime_t * runtime,
    //   device_t * device,
    //   task_t * task,
    //   command_queue_t * iqueue,
    //   command+t * cmd
    //   command_list_counter_t idx
    void (*launch)();
    void * runtime;     // TODO: this should be known implicitly, currently dupplicating on all prog instr :/
    void * device;      // TODO: this should be known implicitly, currently dupplicating on all prog instr :/
    void * task;
};

/* read/write files */
struct command_file_t
{
    int fd;
    void * buffer;
    size_t size;
    size_t offset;
};

/* a batch of multiple dependent commands, contracted by a driver into a single
 * opaque executable (e.g. CUgraphExec on CUDA) */
struct command_graph_t;
struct command_batch_t
{
    /* the command graph of that batch */
    command_graph_t * cg;

    /* driver specific handle */
    void * driver_handle;
};

/* commands */
struct command_t
{
    /* the command type */
    command_type_t type;

    /* type-specific info */
    union
    {
        command_prog_t          prog;
        command_prog_launcher_t prog_launcher;
        command_copy_1D_t       copy_1D;
        command_copy_2D_t       copy_2D;
        command_file_t          file;
        command_batch_t         batch;
    };

    command_t(command_type_t type) : type(type) {}
};

OCG_NAMESPACE_END

#endif /* __OPENCG_COMMAND_HPP__ */
