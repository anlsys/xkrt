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

XKRT_NAMESPACE_USE;

driver_t *
runtime_t::driver_get(
    const driver_type_t type
) {
    assert(type >= 0);
    assert(type < XKRT_DRIVER_TYPE_MAX);
    return this->drivers.list[type];
}

device_t *
runtime_t::device_get(
    const device_unique_id_t device_unique_id
) {
    assert(device_unique_id >= 0);
    assert(device_unique_id < this->drivers.devices.n);
    return this->drivers.devices.list[device_unique_id];
}

device_unique_id_bitfield_t
runtime_t::devices_get(const driver_type_t type)
{
    driver_t * driver = driver_get(type);
    assert(driver);
    return driver->devices.bitfield;
}

static inline command_queue_type_t
command_type_to_queue_type(
    cgir::command_type_t ctype
) {
    switch (ctype)
    {
        case (cgir::COMMAND_TYPE_PROG):
        case (cgir::COMMAND_TYPE_BATCH):
            return XKRT_QUEUE_TYPE_KERN;

        case (cgir::COMMAND_TYPE_COPY_H2H_1D):
        case (cgir::COMMAND_TYPE_COPY_H2H_2D):
            return XKRT_QUEUE_TYPE_H2D;

        case (cgir::COMMAND_TYPE_COPY_H2D_1D):
        case (cgir::COMMAND_TYPE_COPY_H2D_2D):
            return XKRT_QUEUE_TYPE_H2D;

        case (cgir::COMMAND_TYPE_COPY_D2H_1D):
        case (cgir::COMMAND_TYPE_COPY_D2H_2D):
            return XKRT_QUEUE_TYPE_D2H;

        case (cgir::COMMAND_TYPE_COPY_D2D_1D):
        case (cgir::COMMAND_TYPE_COPY_D2D_2D):
            return XKRT_QUEUE_TYPE_D2D;

        case (cgir::COMMAND_TYPE_FD_READ):
            return XKRT_QUEUE_TYPE_FD_READ;

        case (cgir::COMMAND_TYPE_FD_WRITE):
            return XKRT_QUEUE_TYPE_FD_WRITE;

        default:
            LOGGER_FATAL("I don't know what queue I should use for that command!");
            return XKRT_QUEUE_TYPE_ALL;
    }
}

/*
 * Launch a host PROG command according to its launch mode. The launcher is the
 * variadic `void(void**)` function; how it is invoked (and when the command
 * completes) depends on `prog.launch_mode`:
 *
 *   - DIRECT     : the calling thread runs the launcher and the command
 *                  completes as soon as it returns.
 *   - TASK_SPAWN : the runtime spawns a task in the calling thread's team that
 *                  runs the launcher; the command completes when that task
 *                  completes. This is how OpenMP outlined task bodies are
 *                  replayed (the task-spawn used to be baked into the launcher
 *                  function itself, which prevented CGIR from fusing them).
 *
 * In both cases the command's completion callback is raised once the program
 * has run, driving the command-graph replay forward.
 */
static inline void
command_prog_launch_host(
    runtime_t * runtime,
    command_t * command
) {
    assert(command->type == cgir::COMMAND_TYPE_PROG);

    typedef void (*prog_fn_t)(void **);
    prog_fn_t fn   = (prog_fn_t) command->prog.launcher.variadic.fn;
    void **   args = (void **)   command->prog.launcher.variadic.args;
    assert(fn);

    switch (command->prog.launch_mode)
    {
        case (cgir::CGIR_COMMAND_PROG_LAUNCH_MODE_DIRECT):
        {
            fn(args);
            command->completion_callback_raise();
            break ;
        }

        case (cgir::CGIR_COMMAND_PROG_LAUNCH_MODE_TASK_SPAWN):
        {
            runtime->task_spawn<TASK_FLAG_ZERO>(
                (const device_unique_id_t)    XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID,
                (const task_access_counter_t) 0,
                (const task_accesses_setter_t) nullptr,
                (const task_accesses_setter_t) nullptr,
                [fn, args, command] (runtime_t *, device_t *, task_t *) {
                    fn(args);
                    command->completion_callback_raise();   // complete command after the task ran
                }
            );
            break ;
        }

        default:
            LOGGER_FATAL("Unknown PROG launch mode %d", (int) command->prog.launch_mode);
    }
}

int
runtime_t::command_submit(
    const device_unique_id_t device_unique_id,
    command_t * command
) {
    device_t * device = this->device_get(device_unique_id);
    assert(device);

    // command is serialized: it must be launched serially on the calling
    // thread, which returns on the command completion
    if (command->flags & COMMAND_FLAG_SERIALIZED)
    {
        LOGGER_DEBUG("Submitting a serialized command of type `%s`",
                command_type_to_str(command->type));

        // currently only support both serialized and synchronous
        assert(command->flags & COMMAND_FLAG_SYNCHRONOUS);

        switch (command->type)
        {
            case (cgir::COMMAND_TYPE_PROG):
            {
                if (device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID)
                {
                    command_prog_launch_host(this, command);
                    break ;
                }
                // intentionally fallthrough
            }
            case (cgir::COMMAND_TYPE_COPY_H2D_1D):
            case (cgir::COMMAND_TYPE_COPY_D2H_1D):
            case (cgir::COMMAND_TYPE_COPY_D2D_1D):
            case (cgir::COMMAND_TYPE_COPY_H2H_2D):
            case (cgir::COMMAND_TYPE_COPY_H2D_2D):
            case (cgir::COMMAND_TYPE_COPY_D2H_2D):
            case (cgir::COMMAND_TYPE_COPY_D2D_2D):
            case (cgir::COMMAND_TYPE_BATCH):
            {
                driver_t * driver = this->driver_get(device->driver_type);
                assert(driver);

                if (driver->f_command_execute == NULL)
                    LOGGER_FATAL("Not supported");
                driver->f_command_execute(device->driver_id, command);
                break ;
            }

            default:
                LOGGER_FATAL("Unsupported command");
        }
    }
    // else, push to a queue
    else
    {
        command_queue_type_t qtype = command_type_to_queue_type(command->type);

        thread_t * thread;
        command_queue_t * queue;
        device->offloader_queue_next(qtype, &thread, &queue);
        assert(thread);
        assert(queue);

        REENTRANT_SPINLOCK_LOCK(queue->reentrant_spinlock);
        {
            queue->emplace(command);
            queue->commit(command);
        }
        REENTRANT_SPINLOCK_UNLOCK(queue->reentrant_spinlock);

        thread->wakeup();
    }

    return 0;
}
