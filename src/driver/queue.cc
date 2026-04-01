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

//////////////
//  HELPERS //
//////////////

const char *
command_queue_type_to_str(xkrt_command_queue_type_t type)
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

static inline void
command_queue_list_init(
    command_queue_list_t * list,
    uint8_t * buffer,
    xkrt_command_queue_list_counter_t capacity
) {
    list->cmd = (command_t *) buffer;
    list->capacity = capacity;
    list->pos.r = 0;
    list->pos.w = 0;
}

void
command_queue_init(
    command_queue_t * queue,
    command_queue_type_t type,
    xkrt_command_queue_list_counter_t capacity
) {
    queue->type = type;

    uint8_t * mem = (uint8_t *) malloc(
        sizeof(command_t) * capacity    // ready
      + sizeof(command_t) * capacity    // pending
      + sizeof(bool)      * capacity    // completed
    );
    assert(mem);

    command_queue_list_init(
        &queue->ready,
        mem,
        capacity
    );

    command_queue_list_init(
        &queue->pending,
        mem + sizeof(command_t) * capacity,
        capacity
    );

    queue->completed = (bool *) (mem + 2 * sizeof(command_t) * capacity);
    // memset(queue->completed, 0, sizeof(bool) * capacity);
    assert(queue->completed);

    queue->spinlock = SPINLOCK_INITIALIZER;

    # if XKRT_SUPPORT_STATS
    memset(&(queue->stats), 0, sizeof(queue->stats));
    # endif /* XKRT_SUPPORT_STATS */
}

void
command_queue_deinit(command_queue_t * queue)
{
    assert(queue);
    assert(queue->ready.cmd);
    assert(queue->pending.cmd);

    free(queue->ready.cmd);
}

//////////////////////////////////////////
//  MANAGEMENT CALLED BY ANY THREADS    //
//////////////////////////////////////////

command_t *
command_queue_t::command_new(
    const ocg::command_type_t ctype,
    const command_flag_t flags
) {
    if (this->ready.is_full())
        return NULL;

    const xkrt_command_queue_list_counter_t p = this->ready.pos.w;
    assert(0 <= p && p < this->ready.capacity);
    command_t * cmd = this->ready.cmd + p;
    new (cmd) command_t(ctype, flags);

    return cmd;
}

int
command_queue_t::emplace(const command_t * cmd)
{
    memcpy(
        (void *) (this->ready.cmd + this->ready.pos.w),
        (void *) (cmd),
        sizeof(command_t)
    );
    return 0;
}

int
command_queue_t::commit(const command_t * cmd)
{
    // TODO: multiple thread may commit in parallel

    assert(cmd);
    assert(!this->ready.is_full());

    const xkrt_command_queue_list_counter_t p = this->ready.pos.w;
    this->completed[p] = false;
    this->ready.pos.w = (this->ready.pos.w + 1) % this->ready.capacity;
    XKRT_STATS_INCR(this->stats.commands[cmd->type].commited, 1);
    LOGGER_DEBUG(
        "Commited a command of type `%s` (%d ready, %d pending)`",
        ocg::command_type_to_str(cmd->type),
        this->ready.size(),
        this->pending.size()
    );

    return 0;
}

//////////////////////////////////////////////
//  MANAGEMENT CALLED BY THE OWNING THREAD  //
//////////////////////////////////////////////

static inline void
__complete_command_internal(
    command_queue_t * queue,
    const xkrt_command_queue_list_counter_t p
) {
    assert(0 <= p && p < queue->pending.capacity);

    command_t * cmd = queue->pending.cmd + p;
    assert(cmd >= queue->pending.cmd);
    assert(cmd <  queue->pending.cmd + queue->pending.capacity);

    LOGGER_DEBUG(
        "Completed command `%s` on queue %p of type `%s`",
        ocg::command_type_to_str(cmd->type),
        queue,
        command_queue_type_to_str(queue->type)
    );

    queue->completed[p] = true;
    cmd->completion_callback_raise();
    XKRT_STATS_INCR(queue->stats.commands[cmd->type].completed, 1);

    switch (cmd->type)
    {
        case (ocg::COMMAND_TYPE_COPY_H2H_1D):
        case (ocg::COMMAND_TYPE_COPY_H2D_1D):
        case (ocg::COMMAND_TYPE_COPY_D2H_1D):
        case (ocg::COMMAND_TYPE_COPY_D2D_1D):
        {
            XKRT_STATS_INCR(queue->stats.transfered, cmd->copy_1D.size);
            break ;
        }

        case (ocg::COMMAND_TYPE_COPY_H2H_2D):
        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
        {
            XKRT_STATS_INCR(queue->stats.transfered, cmd->copy_2D.m * cmd->copy_2D.n * cmd->copy_2D.sizeof_type);
            break ;
        }

        default:
        {
            break ;
        }
    }
}

void
command_queue_t::complete_command(const xkrt_command_queue_list_counter_t p)
{
    assert(p >= 0);
    assert(p <  this->pending.capacity);
    __complete_command_internal(this, p);
}

void
command_queue_t::complete_commands(const xkrt_command_queue_list_counter_t p)
{
    this->pending.iterate([this] (xkrt_command_queue_list_counter_t p) {
        __complete_command_internal(this, p);
        return true;
    });
    this->pending.pos.r = p;
}

xkrt_command_queue_list_counter_t
command_queue_t::progress(
    const std::function<bool(command_t * cmd, xkrt_command_queue_list_counter_t p)> & process
) {
    return this->pending.iterate([&] (xkrt_command_queue_list_counter_t p) {
        if (this->completed[p])
            return true;
        return process(this->pending.cmd + p, p);
    });
}

XKRT_NAMESPACE_END;
