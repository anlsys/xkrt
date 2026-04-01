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

// This file nd OpenCG data structures for XKRT needs

#ifndef __XKRT_COMMAND_H__
# define __XKRT_COMMAND_H__

# include <opencg/opencg.hpp>

# include <xkrt/callback.h>
# include <xkrt/namespace.h>
# include <xkrt/data-structures/memory-pool.h>
# include <xkrt/sync/mutex.h>
# include <xkrt/sync/spinlock.h>
# include <xkrt/types.h>

XKRT_NAMESPACE_BEGIN;

/* Flags of the command */
typedef enum    command_flag_t
{
    COMMAND_FLAG_NONE           = (0),

    /* if the command should be executed synchronously (i.e., blocking the
     * calling thread until completion) */
    COMMAND_FLAG_SYNCHRONOUS    = (1 << 0),

    /* if the command should be serialized (i.e., so that the thread that
     * emitted it must launch it) */
    COMMAND_FLAG_SERIALIZED     = (1 << 1)

}               command_flag_t;

inline constexpr command_flag_t
operator|(command_flag_t a, command_flag_t b)
{
    return static_cast<command_flag_t>(
        static_cast<int>(a) | static_cast<int>(b)
    );
}

/* commands */
struct command_t : ocg::command_t
{
    /* flags of the command */
    command_flag_t flags;

    /* callback on completion */
    struct {
        callback_t list[XKRT_COMMAND_CALLBACKS_MAX];
        command_callback_index_t n;
    } callbacks;

    /* constructor */
    command_t(
        ocg::command_type_t type,
        command_flag_t flags
    ) :
        ocg::command_t(type),
        flags(flags),
        callbacks{}
    {}

    inline void
    completion_callback_push(const callback_t & callback)
    {
        assert(this->callbacks.n >= 0);
        assert(this->callbacks.n < XKRT_COMMAND_CALLBACKS_MAX);
        this->callbacks.list[this->callbacks.n++] = callback;
    }

    inline void
    completion_callback_clear(void)
    {
        this->callbacks.n = 0;
    }

    inline void
    completion_callback_raise(void)
    {
        for (command_callback_index_t i = 0 ; i < this->callbacks.n ; ++i)
        {
            assert(this->callbacks.list[i].func);
            this->callbacks.list[i].raise();
        }
    }
};

/* Replay counter */
typedef int command_graph_rc_type_t;
typedef std::atomic<command_graph_rc_type_t> command_graph_rc_t;

/* Wait counter of a command node */
typedef int command_graph_node_wc_type_t;
typedef std::atomic<command_graph_node_wc_type_t> command_graph_node_wc_t;

/* node states */
enum command_graph_node_state_t
{
    COMMAND_GRAPH_NODE_STATE_INIT,
    COMMAND_GRAPH_NODE_STATE_COMPLETE
};

/* a node */
struct command_graph_node_t : ocg::command_graph_node_t
{
    /* replay counter */
    command_graph_rc_t rc;

    /* wait counter to know whether this node can be replayed */
    command_graph_node_wc_t wc;

    /* node state */
    command_graph_node_state_t state;

    /* spinlock to reinitialize the node on replay */
    spinlock_t spinlock;

    /* constructor/destructor */
    command_graph_node_t(
        const ocg::command_graph_node_type_t type,
        ocg::command_t * command,
        const ocg::device_unique_id_t device_unique_id
    ) :
        ocg::command_graph_node_t(type, command, device_unique_id),
        rc(0),
        wc(0),
        state(COMMAND_GRAPH_NODE_STATE_INIT),
        spinlock(0)
    {}
};

/* Additional storage for the `command_graph_t` type */
struct command_graph_t : ocg::command_graph_t
{
    /* command allocator */
    memory_pool_t<command_t> commands;

    /* nodes allocator */
    memory_pool_t<command_graph_node_t> nodes;

    /* replay counter */
    command_graph_rc_type_t rc;

    /* mutex/cond to notify threads waiting on the replay completion */
    pthread_mutex_t wait_mtx;
    pthread_cond_t wait_cond;

    command_graph_t(void) :
        commands(),
        nodes(),
        rc(0),
        wait_mtx(),
        wait_cond()
    {}

    /* return number of nodes */
    inline size_t
    get_number_of_allocated_nodes(void) const
    {
        return this->nodes.size();
    }

};

XKRT_NAMESPACE_END;

#endif /* __XKRT_COMMAND_H__ */
