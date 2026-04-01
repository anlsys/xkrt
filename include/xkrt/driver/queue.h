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

#ifndef __QUEUE_HPP__
# define __QUEUE_HPP__

# include <xkrt/command/command.hpp>
# include <xkrt/driver/queue-type.h>
# include <xkrt/stats/stats.h>
# include <xkrt/support.h>
# include <xkrt/sync/lockable.hpp>
# include <xkrt/thread/thread.h>

# include <atomic>

XKRT_NAMESPACE_BEGIN

const char * command_queue_type_to_str(xkrt_command_queue_type_t type);

struct command_queue_list_t
{
    // TODO: memory layout: do we want AoS or SoA here ?
    command_t * cmd;                                 /* commands buffer */
    xkrt_command_queue_list_counter_t capacity;      /* buffer capacity */
    struct {
        volatile xkrt_command_queue_list_counter_t r; /* first command to process */
        volatile xkrt_command_queue_list_counter_t w; /* next position for inserting commands */
    } pos;

    /* methods */
    int
    is_full(void) const
    {
        return (this->pos.w  == this->pos.r - 1);
    }

    int
    is_empty(void) const
    {
        return (this->pos.r == this->pos.w);
    }

    xkrt_command_queue_list_counter_t
    size(void) const
    {
        if (this->pos.r <= this->pos.w)
            return (this->pos.w - this->pos.r);
        else
            return this->capacity - this->pos.r + this->pos.w;
    }

    /**
     *  Iterate on each command at index p of the list,
     *  and stop early if process(p) returns false
     */
    inline xkrt_command_queue_list_counter_t
    iterate(const std::function<bool(xkrt_command_queue_list_counter_t p)> & process)
    {
        const xkrt_command_queue_list_counter_t a = this->pos.r;
        const xkrt_command_queue_list_counter_t b = this->pos.w;

        assert(a < this->capacity);
        assert(b < this->capacity);

        if (a <= b) {
            for (xkrt_command_queue_list_counter_t i = a; i < b; ++i)
                if (!process(i)) return i;
        } else {
            for (xkrt_command_queue_list_counter_t i = a; i < capacity; ++i)
                if (!process(i)) return i;
            for (xkrt_command_queue_list_counter_t i = 0; i < b; ++i)
                if (!process(i)) return i;
        }
        return b;
    }
};

/* this is a 'io_queue' equivalent */
struct command_queue_t
{
    /* the type of that queue */
    command_queue_type_t type;

    // TODO: currently, ready/pending/completed are SoA, we probably want AoS to:
    //  - perform fast copy from ready to pending
    //  - fast test of completion given a command

    /* queue for ready command */
    command_queue_list_t ready;

    /* queue for pending commands to progress */
    command_queue_list_t pending;

    /* whether command at index 'p' is completed */
    bool * completed;

    /* spinlock on the ready queue
     *  - any threasd may push to it
     *  - the owning thread may move from it to the pending queue
     */
    spinlock_t spinlock;

    # if XKRT_SUPPORT_STATS
    struct {
        struct {
            stats_int_t commited;
            stats_int_t completed;
        } commands[ocg::COMMAND_TYPE_MAX];
        stats_int_t transfered;
    } stats;
    # endif /* XKRT_SUPPORT_STATS */

    /**
     *  Return true if the queue is full of commands, false otherwise
     *  Threading: called by any thread
     */
    int is_full(void) const;

    /**
     *  Allocate a new command to the queue (must then be commited via 'commit') later
     *  Threading: called by any thread
     */
    command_t * command_new(const ocg::command_type_t ctype, const command_flag_t flags);

    /**
     *  Commit a command previously allocated via 'command_new'
     *  Threading: called by any thread
     */
    int commit(const command_t * command);

    /**
     *  Memcpy the passed command to the end of the ready queue.
     *  Threading: called by any thread
     */
    int emplace(const command_t * command);

    /**
     *  Iterate on each command at index p of the list, if it is not completed already.
     *  Stop early if process(cmd, p) returned false
     *  Threading: called by the owning thread only
     */
    xkrt_command_queue_list_counter_t progress(const std::function<bool(command_t * cmd, xkrt_command_queue_list_counter_t p)> & process);

    /**
     *  Complete the command at the i-th position in the pending queue (invoke callbacks)
     *  Threading: called by the owning thread only
     */
    void complete_command(const xkrt_command_queue_list_counter_t p);

    /**
     *  Complete all commands until index 'ok_p' (see complete_command)
     *  Threading: called by the owning thread only
     */
    void complete_commands(const xkrt_command_queue_list_counter_t ok_p);

};  /* command_queue_t */

void command_queue_init(
    command_queue_t * queue,
    command_queue_type_t qtype,
    xkrt_command_queue_list_counter_t capacity
);

void command_queue_deinit(command_queue_t * queue);

XKRT_NAMESPACE_END

#endif /* __QUEUE_HPP__ */
