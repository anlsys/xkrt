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

# include <xkrt/runtime.h>
# include <xkrt/sync/atomic.h>

XKRT_NAMESPACE_USE;

////////////////
// ALLOCATORS //
////////////////

static command_t *
command_new(
    command_graph_t * cg,
    ocg::command_type_t type
) {
    return cg->commands.put(type, COMMAND_FLAG_NONE);
}

static command_graph_node_t *
command_graph_node_new(
    command_graph_t * cg,
    command_t * command,
    const ocg::device_unique_id_t device_unique_id
) {
    static_assert(sizeof(command_graph_node_t) <= sizeof(command_graph_node_t));
    return cg->nodes.put(command, device_unique_id);
}

void command_graph_init(command_graph_t * cg);

static command_graph_t *
command_graph_new(command_graph_t * original_cg)
{
    command_graph_t * cg = (command_graph_t *) malloc(sizeof(command_graph_t));
    assert(cg);
    command_graph_init(cg);
    return cg;
}

void
command_graph_init(command_graph_t * cg)
{
    new (cg) command_graph_t();
    cg->init(
        (ocg::command_allocator_t)              command_new,
        (ocg::command_graph_node_allocator_t)   command_graph_node_new,
        (ocg::command_graph_allocator_t)        command_graph_new
    );
}

//////////////////////////////
// CONSTRUCT FROM TASKGRAPH //
//////////////////////////////

// Each task get 4 control nodes:
// - N1 - the entry node is the root of the task sub-command graph
// - N3 - is the fetching state
// - N5 - is the executing state
// - N7 - is the completed state
// and 1 node per emitted command

void
runtime_t::command_graph_from_task_dependency_graph(
    task_dependency_graph_t * tdg,              /* IN  */
    command_graph_t * cg                        /* OUT */
) {
    // base get_number_of_allocated_nodes
    const size_t ntasks = tdg->get_ntasks();

    // TODO: we should save graph complexity here: if there is no emitted
    // commands for a state, then the control node can be skipped

    // TODO: we could save this malloc, by hiting directly in the cg->nodes struct.
    // Each task is getting 4 control nodes pushed to the cg, offset by 2 (entry/exit of the cg)

    // allocate a temporary buffer to store N1, N3, N5 and N7 of each task
    # define N_CONTROL_NODES_PER_TASK 4
    command_graph_node_t ** control_nodes = (command_graph_node_t **) malloc(sizeof(command_graph_node_t *) * ntasks * N_CONTROL_NODES_PER_TASK);
    assert(control_nodes);

    // get entry/exit nodes
    command_graph_init(cg);
    command_graph_node_t * entry = (command_graph_node_t *) cg->node_get_entry();
    command_graph_node_t * exit  = (command_graph_node_t *) cg->node_get_exit();
    assert(entry);
    assert(exit);

    // iterate to instanciate and connect nodes
    tdg->foreach_task([&] (task_t * task)
    {
        // get device on which the task executed - we reexecute on the same device
        const device_unique_id_t device_unique_id = task_get_device_unique_id(task);

        // Generate task sub-cg
        task_rec_info_t * taskrec = TASK_REC_INFO(task);

        //  1) empty node N1 (entry of that task subcg)
        //  2)  commands N2i emmited during fetching
        //  3) empty node N3 that depend on all N1i
        //  4)  commands N4i emitted during routine execution that depend on N2
        //  5) empty node N5 that depends on all N3i
        //  6)  commands N6i emitted after completion, for prefetching
        //  7) empty node N7 that depends on all N5i (sink of the task cg)

        command_graph_node_t * N1 = command_graph_node_new(cg, NULL, device_unique_id);
        command_graph_node_t * N3 = command_graph_node_new(cg, NULL, device_unique_id);
        command_graph_node_t * N5 = command_graph_node_new(cg, NULL, device_unique_id);
        command_graph_node_t * N7 = command_graph_node_new(cg, NULL, device_unique_id);

        assert(taskrec->index < ntasks);
        control_nodes[taskrec->index * N_CONTROL_NODES_PER_TASK + 0] = N1;
        control_nodes[taskrec->index * N_CONTROL_NODES_PER_TASK + 1] = N3;
        control_nodes[taskrec->index * N_CONTROL_NODES_PER_TASK + 2] = N5;
        control_nodes[taskrec->index * N_CONTROL_NODES_PER_TASK + 3] = N7;

        // to track if all commands were emitted on the same device
        device_unique_id_t prev_cmd_device_unique_id = XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID;
        bool all_cmd_are_on_same_device = true;

        // add commands emitted by the task
        for (task_command_record_t & rec : taskrec->commands)
        {
            // sanity check: commands may only be emmited from fetching
            // accesses, or running task routine
            assert(rec.state == TASK_STATE_DATA_FETCHING || rec.state == TASK_STATE_EXECUTING || rec.state == TASK_STATE_COMPLETED);

            // maybe update the `device_unique_id`
            device_unique_id_t cmd_device_unique_id = device_unique_id;
            switch (rec.command.type)
            {
                // It is possible that the task was schedule on a host thread,
                // and spawned device commands that were recorded (e.g., mcc coherence, or prefetching)
                // We want to replay them on the implicit team of threads of the actual device: not on the host team.

                case (ocg::COMMAND_TYPE_COPY_H2D_1D):
                {
                    cmd_device_unique_id = rec.command.copy_1D.dst_device_unique_id;
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_1D):
                case (ocg::COMMAND_TYPE_COPY_D2D_1D):
                {
                    cmd_device_unique_id = rec.command.copy_1D.src_device_unique_id;
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_H2D_2D):
                {
                    cmd_device_unique_id = rec.command.copy_2D.dst_device_unique_id;
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_2D):
                case (ocg::COMMAND_TYPE_COPY_D2D_2D):
                {
                    cmd_device_unique_id = rec.command.copy_2D.src_device_unique_id;
                    break ;
                }

                default:
                    break ;
            }

            // track if all cmd were emitted to the same device
            if (prev_cmd_device_unique_id == XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID)
            {
                prev_cmd_device_unique_id = cmd_device_unique_id;
            }
            else if (!all_cmd_are_on_same_device && prev_cmd_device_unique_id != cmd_device_unique_id)
            {
                all_cmd_are_on_same_device = false;
            }

            // Create a node
            command_graph_node_t * N = command_graph_node_new(cg, &rec.command, cmd_device_unique_id);
            assert(N);

            // link it in the command graph
            switch (rec.state)
            {
                case (TASK_STATE_DATA_FETCHING):
                {
                    N1->precedes(N);
                    N ->precedes(N3);
                    break ;
                }

                case (TASK_STATE_EXECUTING):
                {
                    N3->precedes(N);
                    N ->precedes(N5);
                    break ;
                }

                case (TASK_STATE_COMPLETED):
                {
                    N5->precedes(N);
                    N ->precedes(N7);
                    break ;
                }

                default:
                    LOGGER_FATAL("Not supported");
            }
        }

        // if no commands were emitted during the fetching state,
        // be sure source/sink of that task are connected
        if (N1->successors.size() == 0)
            N1->precedes(N3);

        // if no commands were emitted during the executing state,
        // be sure source/sink of that task are connected
        if (N3->successors.size() == 0)
            N3->precedes(N5);

        // if no commands were emitted for prefetching,
        // be sure source/sink of that task are connected
        if (N5->successors.size() == 0)
            N5->precedes(N7);

        // if all commands occured on the same device,
        // rewrite device of control nodes
        if (all_cmd_are_on_same_device && prev_cmd_device_unique_id != device_unique_id)
        {
            N1->device_unique_id = prev_cmd_device_unique_id;
            N3->device_unique_id = prev_cmd_device_unique_id;
            N5->device_unique_id = prev_cmd_device_unique_id;
            N7->device_unique_id = prev_cmd_device_unique_id;
        }
    });

    // iterate through each tasks to connect sub-cgs, so that:
    tdg->foreach_task([&] (task_t * task)
    {
        task_rec_info_t * rec = TASK_REC_INFO(task);
        command_graph_node_t * N1 = control_nodes[rec->index * N_CONTROL_NODES_PER_TASK + 0];

        //  if T1 -> T2, then exit(T1) -> entry(T2)
        for (access_t * pred_access : rec->predecessors)
        {
            task_t * pred = pred_access->task;
            task_rec_info_t * pred_rec = TASK_REC_INFO(pred);
            command_graph_node_t * N7 = control_nodes[pred_rec->index * N_CONTROL_NODES_PER_TASK + 3];
            N7->precedes(N1);
        }
    });

    // iterate a last time to connect to entry/exit
    tdg->foreach_task([&] (task_t * task)
    {
        task_rec_info_t * rec = TASK_REC_INFO(task);
        command_graph_node_t * N1 = control_nodes[rec->index * N_CONTROL_NODES_PER_TASK + 0];

        // if N1 has no predecessor, then entry -> N1
        if (N1->predecessors.size() == 0)
            entry->precedes(N1);

        // if N7 has no successor, then N7 -> exit
        command_graph_node_t * N7 = control_nodes[rec->index * N_CONTROL_NODES_PER_TASK + 3];
        if (N7->successors.size() == 0)
            N7->precedes(exit);
    });

    // release control nodes buffer
    free(control_nodes);
}

////////////
// REPLAY //
////////////

void command_graph_replay_node_complete(
    runtime_t * runtime,
    command_graph_t * cg,
    command_graph_node_t * node
);

static void
command_graph_replay_node_completion_callback(
    void * args[XKRT_CALLBACK_ARGS_MAX]
) {
    runtime_t            * runtime = (runtime_t *)            args[0];
    command_graph_t      * cg      = (command_graph_t *)      args[1];
    command_graph_node_t * node    = (command_graph_node_t *) args[2];
    return command_graph_replay_node_complete(runtime, cg, node);
}

template <command_graph_node_wc_type_t initial_dwc>
static inline void
command_graph_replay_process_node(
    runtime_t * runtime,
    command_graph_t * cg,
    command_graph_node_t * node
) {
    // delta wait counter
    command_graph_node_wc_type_t dwc = initial_dwc;

    SPINLOCK_LOCK(node->spinlock);
    {
        // first time processing this node for that replay: reinitialize the node
        if (xkrt_compare_and_swap(node->rc, cg->rc))
        {
            if (node->command)
            {
                // clear all previous callbacks
                ((command_t *) node->command)->completion_callback_clear();

                // add a callback to notify successor commands of that predecessor completion
                static_assert(XKRT_CALLBACK_ARGS_MAX >= 3);
                callback_t callback;
                callback.func = command_graph_replay_node_completion_callback;
                callback.args[0] = runtime;
                callback.args[1] = cg;
                callback.args[2] = node;
                ((command_t *) node->command)->completion_callback_push(callback);
            }

            // set node in INIT state
            node->state = COMMAND_GRAPH_NODE_STATE_INIT;

            // increment wc to avoid early release
            static_assert(std::is_signed<command_graph_node_wc_type_t>::value,
                    "command_graph_node_wc_t must be able to represent negative numbers");
            dwc -= (command_graph_node_wc_type_t) node->predecessors.size();
        }
    }
    SPINLOCK_UNLOCK(node->spinlock);

    // if we reached 0, command is now ready
    if (node->wc.fetch_sub(dwc, std::memory_order_relaxed) == dwc)
    {
        // if the node holds a command
        if (node->command)
        {
            assert(node->device_unique_id != XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID);
            assert(node->state == COMMAND_GRAPH_NODE_STATE_INIT);
            runtime->command_submit(node->device_unique_id, (command_t *) node->command);
        }
        // else, submit its descendence
        else
        {
            // completes it without submitting any command
            command_graph_replay_node_complete(runtime, cg, node);
        }
    }
}

void
command_graph_replay_node_complete(
    runtime_t * runtime,
    command_graph_t * cg,
    command_graph_node_t * node
) {
    // submit successor nodes
    node->foreach_successor([&] (ocg::command_graph_node_t * succ) {
        command_graph_replay_process_node<1>(runtime, cg, (command_graph_node_t *) succ);
    });

    assert(node->state == COMMAND_GRAPH_NODE_STATE_INIT);

    // we completed the last node, notify waiting threads
    if (node == cg->node_get_exit())
    {
        pthread_mutex_lock(&cg->wait_mtx);
        {
            node->state = COMMAND_GRAPH_NODE_STATE_COMPLETE;
            pthread_cond_signal(&cg->wait_cond);
        }
        pthread_mutex_unlock(&cg->wait_mtx);
    }
    else
    {
        // we don't care about that state change in such case
        // node->state = COMMAND_GRAPH_NODE_STATE_COMPLETE;
    }
}

void
runtime_t::command_graph_replay(command_graph_t * cg)
{
    // increase replay counter
    ++cg->rc;

    // get entry/exit nodes to launch and wait completion
    command_graph_node_t * entry = (command_graph_node_t *) cg->node_get_entry();
    command_graph_node_t * exit  = (command_graph_node_t *) cg->node_get_exit();

    // submit and reinitialized entry and exit nodes.
    // Exit must be initialized first, to avoid early completion
    command_graph_replay_process_node<0>(this, cg, exit);
    command_graph_replay_process_node<0>(this, cg, entry);

    // wait for exit completion
    pthread_mutex_lock(&cg->wait_mtx);
    {
        while ((volatile command_graph_node_state_t) exit->state != COMMAND_GRAPH_NODE_STATE_COMPLETE)
            pthread_cond_wait(&cg->wait_cond, &cg->wait_mtx);
    }
    pthread_mutex_unlock(&cg->wait_mtx);
}

/////////////
// DESTROY //
/////////////

void
runtime_t::command_graph_destroy(command_graph_t * cg)
{
    cg->nodes.release();
    cg->commands.release();
}
