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
    ocg::command_type_t ctype
) {
    switch (ctype)
    {
        case (ocg::COMMAND_TYPE_PROG):
        case (ocg::COMMAND_TYPE_BATCH):
            return XKRT_QUEUE_TYPE_KERN;

        case (ocg::COMMAND_TYPE_COPY_H2H_1D):
        case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            return XKRT_QUEUE_TYPE_H2D;

        case (ocg::COMMAND_TYPE_COPY_H2D_1D):
        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            return XKRT_QUEUE_TYPE_H2D;

        case (ocg::COMMAND_TYPE_COPY_D2H_1D):
        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            return XKRT_QUEUE_TYPE_D2H;

        case (ocg::COMMAND_TYPE_COPY_D2D_1D):
        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
            return XKRT_QUEUE_TYPE_D2D;

        case (ocg::COMMAND_TYPE_FD_READ):
            return XKRT_QUEUE_TYPE_FD_READ;

        case (ocg::COMMAND_TYPE_FD_WRITE):
            return XKRT_QUEUE_TYPE_FD_WRITE;

        default:
            LOGGER_FATAL("I don't know what queue I should use for that command!");
            return XKRT_QUEUE_TYPE_ALL;
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
        // currently only support both serialized and synchronous
        assert(command->flags & COMMAND_FLAG_SYNCHRONOUS);

        switch (command->type)
        {
            case (ocg::COMMAND_TYPE_PROG):
            {
                if (device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID)
                    command->prog.launcher.fixed.fn(command->prog.launcher.fixed.args);
                else
                    LOGGER_FATAL("Unsupported command for non-host device");
                break ;
            }

            case (ocg::COMMAND_TYPE_COPY_H2D_1D):
            case (ocg::COMMAND_TYPE_COPY_D2H_1D):
            case (ocg::COMMAND_TYPE_COPY_D2D_1D):
            case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            case (ocg::COMMAND_TYPE_COPY_D2D_2D):
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

        SPINLOCK_LOCK(queue->spinlock);
        queue->emplace(command);
        queue->commit(command);
        SPINLOCK_UNLOCK(queue->spinlock);

        thread->wakeup();
    }

    return 0;
}
