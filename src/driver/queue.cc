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

# include <errno.h>
# include <string.h>

# include <xkrt/logger/logger.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/todo.h>

XKRT_NAMESPACE_BEGIN;

const char *
queue_type_to_str(xkrt_queue_type_t type)
{
    switch (type)
    {
        case (XKRT_QUEUE_TYPE_H2D):        return "H2D";
        case (XKRT_QUEUE_TYPE_D2H):        return "D2H";
        case (XKRT_QUEUE_TYPE_D2D):        return "D2D";
        case (XKRT_QUEUE_TYPE_P2P):        return "P2P";
        case (XKRT_QUEUE_TYPE_KERN):       return "KERN";
        case (XKRT_QUEUE_TYPE_FD_READ):    return "FD_READ";
        case (XKRT_QUEUE_TYPE_FD_WRITE):   return "FD_WRITE";
        case (XKRT_QUEUE_TYPE_ALL):        return "ALL";
        default:                           return  NULL;
    }
}

const char *
command_type_to_str(command_type_t type)
{
     switch (type)
     {
         case (COMMAND_TYPE_KERN):        return "KERN";
         case (COMMAND_TYPE_COPY_H2H_1D): return "COPY_H2H_1D";
         case (COMMAND_TYPE_COPY_H2D_1D): return "COPY_H2D_1D";
         case (COMMAND_TYPE_COPY_D2H_1D): return "COPY_D2H_1D";
         case (COMMAND_TYPE_COPY_D2D_1D): return "COPY_D2D_1D";
         case (COMMAND_TYPE_COPY_H2H_2D): return "COPY_H2H_2D";
         case (COMMAND_TYPE_COPY_H2D_2D): return "COPY_H2D_2D";
         case (COMMAND_TYPE_COPY_D2H_2D): return "COPY_D2H_2D";
         case (COMMAND_TYPE_COPY_D2D_2D): return "COPY_D2D_2D";
         case (COMMAND_TYPE_FD_READ):     return "FD_READ";
         case (COMMAND_TYPE_FD_WRITE):    return "FD_WRITE";
         default:                                   return NULL;
    }
}

static inline void
queue_command_list_init(
    queue_command_list_t * list,
    uint8_t * buffer,
    queue_command_list_counter_t capacity
) {
    list->cmd = (command_t *) buffer;
    list->capacity = capacity;
    list->pos.r = 0;
    list->pos.w = 0;
}

void
queue_init(
    queue_t * queue,
    queue_type_t type,
    queue_command_list_counter_t capacity,
    int (*f_queue_launch)(queue_t * queue, command_t * cmd, queue_command_list_counter_t idx),
    int (*f_queues_progress)(queue_t * queue),
    int (*f_queues_wait)(queue_t * queue),
    int (*f_queue_wait)(queue_t * queue, command_t * cmd, queue_command_list_counter_t idx)
) {
    queue->type = type;
    queue->spinlock = SPINLOCK_INITIALIZER;

    queue->f_command_launch    = f_queue_launch;
    queue->f_commands_progress = f_queues_progress;
    queue->f_commands_wait     = f_queues_wait;
    queue->f_command_wait      = f_queue_wait;

    uint8_t * mem = (uint8_t *) malloc(sizeof(command_t) * capacity * 2);
    assert(mem);

    queue_command_list_init(
        &queue->ready,
        mem,
        capacity
    );

    queue_command_list_init(
        &queue->pending,
        mem + sizeof(command_t) * capacity,
        capacity
    );

    # if XKRT_SUPPORT_STATS
    memset(&(queue->stats), 0, sizeof(queue->stats));
    # endif /* XKRT_SUPPORT_STATS */
}

void
queue_deinit(queue_t * queue)
{
    assert(queue);
    assert(queue->ready.cmd);
    assert(queue->pending.cmd);

    free(queue->ready.cmd);
}

int
queue_t::commit(command_t * cmd)
{
    assert(cmd);
    assert(!this->ready.is_full());

    this->ready.pos.w = (this->ready.pos.w + 1) % this->ready.capacity;
    XKRT_STATS_INCR(this->stats.commands[cmd->type].commited, 1);

    LOGGER_DEBUG(
        "Commited a command of type `%s` (%d ready, %d pending)`",
        command_type_to_str(cmd->type),
        this->ready.size(),
        this->pending.size()
    );

    this->unlock();

    return 0;
}

int
queue_t::launch_ready_commands(void)
{
    assert(this->f_command_launch);
    if (this->ready.is_empty())
        return 0;

    int r = 0;

    /* for each ready command */
    const queue_command_list_counter_t p = this->ready.iterate([this, &r] (queue_command_list_counter_t p) {

        /* if the pending queue is full, we cannot start more commands */
        if (this->pending.is_full())
            return false;

        /* retrieve it */
        command_t * cmd = this->ready.cmd + p;
        assert(cmd);

        LOGGER_DEBUG(
            "Decoding command `%s` on queue %p of type `%s` - p=%u, r=%u, w=%u",
            command_type_to_str(cmd->type),
            this,
            queue_type_to_str(this->type),
            p,
            this->ready.pos.r,
            this->ready.pos.w
        );

        /* launch command */
        switch (cmd->type)
        {
            /* kernel commands are launched by the command itself, not the driver */
            case (COMMAND_TYPE_KERN):
            {
                ((kernel_launcher_t) cmd->kern.launch)(cmd->kern.runtime, cmd->kern.device, cmd->kern.task, this, cmd, p);
                break ;
            }

            case (COMMAND_TYPE_COPY_H2H_1D):
            case (COMMAND_TYPE_COPY_H2H_2D):
            {
                LOGGER_FATAL("Not implemented");
                break ;
            }

            /* launch command */
            case (COMMAND_TYPE_COPY_H2D_1D):
            case (COMMAND_TYPE_COPY_D2H_1D):
            case (COMMAND_TYPE_COPY_D2D_1D):
            case (COMMAND_TYPE_COPY_H2D_2D):
            case (COMMAND_TYPE_COPY_D2H_2D):
            case (COMMAND_TYPE_COPY_D2D_2D):
            case (COMMAND_TYPE_FD_READ):
            case (COMMAND_TYPE_FD_WRITE):
            default:
            {
                int err = this->f_command_launch(this, cmd, p);
                switch (err)
                {
                    case (0):
                    case (EINPROGRESS):
                    {
                        break ;
                    }

                    case (ENOSYS):
                    {
                        LOGGER_FATAL("Command `%s` not implemented", command_type_to_str(cmd->type));
                        break ;
                    }

                    default:
                    {
                        LOGGER_FATAL("Unknown error after decoding command");
                        break ;
                    }
                }
            }
        }

        /* if the command is synchronous, it is completed now */
        if (cmd->synchronous)
        {
            this->complete_command(p);
        }
        /* else, save to pending list */
        else
        {
            /* the pending queue must not be full at that point */
            assert(!this->pending.is_full());
            const queue_command_list_counter_t wp = this->pending.pos.w;
            this->pending.pos.w = (this->pending.pos.w + 1) % this->pending.capacity;

            memcpy(
                (void *) (this->pending.cmd + wp),
                (void *) (this->ready.cmd   + p),
                sizeof(command_t)
            );

            ++r;
        }

        /* continue */
        LOGGER_DEBUG("(loop) ready.is_empty() = %d, pending.is_empty() = %d", this->ready.is_empty(), this->pending.is_empty());
        return true;

    }); /* RING_ITERATE */

    // this barrier ensures that the threads that owns the queue correctly
    // sees the 'ready' queue empty but not the 'pending' - else it would go
    // to sleep even though there is pending commands
    writemem_barrier();
    this->ready.pos.r = p;

    LOGGER_DEBUG("ready.is_empty() = %d, pending.is_empty() = %d", this->ready.is_empty(), this->pending.is_empty());

    return r;
}

// TODO : allow out of order completion

template <bool set_completed_flag>
static inline void
__complete_command_internal(
    queue_t * queue,
    command_t * cmd
) {
    assert(cmd >= queue->pending.cmd);
    assert(cmd <  queue->pending.cmd + queue->pending.capacity);

    LOGGER_DEBUG(
        "Completed command `%s` on queue %p of type `%s`",
        command_type_to_str(cmd->type),
        queue,
        queue_type_to_str(queue->type)
    );

    if (set_completed_flag)
        cmd->completed = true;

    for (command_callback_index_t i = 0 ; i < cmd->callbacks.n ; ++i)
    {
        assert(cmd->callbacks.list[i].func);
        cmd->callbacks.list[i].func(cmd->callbacks.list[i].args);
    }

    XKRT_STATS_INCR(queue->stats.commands[cmd->type].completed, 1);
}

template <bool set_completed_flag>
static inline void
__complete_command_internal(queue_t * queue, const queue_command_list_counter_t p)
{
    command_t * cmd = queue->pending.cmd + p;
    __complete_command_internal<set_completed_flag>(queue, cmd);
}

// complete the given command
void
queue_t::complete_command(const queue_command_list_counter_t p)
{
    assert(p >= 0);
    assert(p <  this->pending.capacity);
    __complete_command_internal<true>(this, p);
}

void
queue_t::complete_command(command_t * cmd)
{
    __complete_command_internal<true>(this, cmd);
}

void
queue_t::complete_commands(const queue_command_list_counter_t p)
{
    this->pending.iterate([this] (queue_command_list_counter_t p) {
        __complete_command_internal<false>(this, p);
        return true;
    });
    this->pending.pos.r = p;
}

void
queue_t::wait_pending_commands(void)
{
    if (!this->pending.is_empty())
    {
        this->f_commands_wait(this);
        this->complete_commands(this->pending.pos.w);
    }
}

int
queue_t::progress_pending_commands(void)
{
    assert(this->f_commands_progress);

    if (this->pending.is_empty())
        return 0;

    LOGGER_DEBUG("Progressing pending commands of queue %p of type `%s` (%d pending) - ptr at r=%u, w=%u",
            this, queue_type_to_str(this->type), this->pending.size(), this->pending.pos.r, this->pending.pos.w);

    // ask for progression of the given commands
    const int r = this->f_commands_progress(this);

    // move reading position to first uncompleted cmd
    const queue_command_list_counter_t p = this->pending.iterate([this] (queue_command_list_counter_t p) {
        return (this->pending.cmd[p].completed) ? true : false;
    });
    this->pending.pos.r = p;

    LOGGER_DEBUG("Progressed pending commands of queue %p of type `%s` (%d pending)",
            this, queue_type_to_str(this->type), this->pending.size());

    // return err code
    return r;
}

XKRT_NAMESPACE_END;

