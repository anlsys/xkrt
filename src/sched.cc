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

# include <xkrt/memory/access/blas/memory-tree.hpp>
# include <xkrt/runtime.h>
# include <xkrt/internals.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/bits-to-str.h>
# include <xkrt/logger/todo.h>
# include <xkrt/sync/mem.h>
# include <xkrt/stats/stats.h>
# include <xkrt/task/task.hpp>

# if XKRT_SUPPORT_JULIA
#  include <julia.h>
# endif

# include <cassert>
# include <cstring>
# include <cerrno>

# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif /* _GNU_SOURCE */
# include <sched.h> /* getcpu */

XKRT_NAMESPACE_BEGIN

/////////////////////////
//  DEVICE PROGRESSION //
/////////////////////////

static inline void
__task_moldable_split(
    runtime_t * runtime,
    task_t * task,
    access_t * accesses,
    task_acs_info_t * acs
) {
    // right now, all task accesses must have the same type
    const access_type_t type = (accesses + 0)->type;

    # if XKRT_SUPPORT_DEBUG
    for (task_access_counter_t i = 1 ; i < acs->ac ; ++i)
        assert(type == (accesses + i)->type);
    # endif /* atm, only support moldable tasks with accesses of same type */

    switch (type)
    {
        case (ACCESS_TYPE_SEGMENT):
        {
            // dupplicate the task
            task_t * dup_task = runtime->task_dup(task);
            assert(dup_task);

            // split accesses and refine dependencies
            access_t * dup_accesses = TASK_ACCESSES(dup_task);
            assert(dup_accesses);

            // for each access
            for (task_access_counter_t i = 0 ; i < acs->ac ; ++i)
            {
                access_t * access     = accesses     + i;
                access_t * dup_access = dup_accesses + i;

                assert(access->task     == task);
                assert(dup_access->task == task);
                dup_access->task = dup_task;
                assert(access->task     == task);
                assert(dup_access->task == dup_task);

                // split access
                access_t::split(access, dup_access, dup_task, ACCESS_SPLIT_MODE_HALVES);

                // for each original successor
                for (access_t * succ_access : access->successors)
                {
                    // if new access conflicts
                    if (access_t::conflicts(dup_access, succ_access))
                    {
                        // set a dependency
                        __access_precedes(dup_access, succ_access);
                    }
                    else
                    {
                        // nothing to do
                    }

                    // if shrinked access still conflicts
                    if (access_t::conflicts(access, succ_access))
                    {
                        // nothing to do, we recycle task and the access,
                        // so the dependency is already set
                    }
                    else
                    {
                        // TODO: unset the dependency
                        LOGGER_FATAL("TODO: should unref once the successor");
                    }
                }
            }

            // submit the dupplicated task
            assert(dup_task->parent);
            assert(dup_task->parent == task->parent);

            # pragma message(TODO "This is quite ugly, can we have tasks go through a more regular transition path ? At that point, the dupplicated task is in the 'ready' state already, as its original task was ready")
            ++dup_task->parent->cc;
            runtime->task_enqueue(dup_task);

            break ;
        }

        default:
            LOGGER_FATAL("Not supported");
    }
}

static inline void
__device_prepare_task(
    runtime_t * runtime,
    device_t * device,
    task_t * task
) {
    assert(device);
    assert(task);
    assert(task->state.value == TASK_STATE_READY);

    /* if that's a device task, then fetches to the device. Else, fetch to the host */
    device_unique_id_t device_unique_id = (task->flags & TASK_FLAG_DEVICE) ? device->unique_id : XKRT_HOST_DEVICE_UNIQUE_ID;
    LOGGER_DEBUG("Preparing task `%s` of format `%d` on device `%d` - on a thread of device `%d`",
            task->label, task->fmtid, device_unique_id, device->unique_id);

    /* if the task has accesses, ensure each of them are coherent before starting execution */
    if (task->flags & TASK_FLAG_ACCESSES)
    {
        task_acs_info_t * acs = TASK_ACS_INFO(task);
        assert(TASK_ACS_INFO(task)->wc == 0);

        /* if there is at least one access */
        if (acs->ac > 0)
        {
            /* retrieve accesses */
            access_t * accesses = TASK_ACCESSES(task);

            /////////////////////////
            // MOLDABLE TASK SPLIT //
            /////////////////////////

            // TODO : move that to another function

            /* if the task is moldable */
            if (task->flags & TASK_FLAG_MOLDABLE)
            {
                /* check split condition, and split, or execute normally */
                task_mol_info_t * mol = TASK_MOL_INFO(task);
                assert(mol->split_condition);

                /* if the moldable task must split */
                if (mol->split_condition(task, accesses))
                {
                    // shrink the moldable task, and resubmit the original task
                    // whose accesses got shrinked
                    __task_moldable_split(runtime, task, accesses, acs);
                    return runtime->task_enqueue(task);
                }
                else
                {
                    // mothing to do
                }
            }
            else
            {
                // nothing to do
            }

            ////////////////////
            // FETCH ACCESSES //
            ////////////////////

            /* increase task 'fetching' counter so it does not get ready early
             * (eg before we processed all accesses bellow) */
            __task_fetching(1, task);

            /* Set 'task' as current task record when fetching, so that emitted commands are recorded to 'task' */
            thread_t * thread = thread_t::get_tls();
            if (task->flags & TASK_FLAG_RECORD) thread->current_task_record = task;

            /* for each access */
            assert(acs->ac <= XKRT_TASK_MAX_ACCESSES);
            for (task_access_counter_t i = 0 ; i < acs->ac ; ++i)
            {
                access_t * access = accesses + i;
                if (access->mode & ACCESS_MODE_V)
                    continue ;

                assert(task == access->task);
                MemoryCoherencyController * mcc = task_get_memory_controller(runtime, task->parent, access);
                if (mcc)
                    mcc->fetch(access, device_unique_id);
            }
            if (task->flags & TASK_FLAG_RECORD) thread->current_task_record = NULL;

            /* decrease the task 'fetching' counter to detect early-fetch completion */
            __task_fetched(1, task, task_execute, runtime, device);
            /* else the task will be launched in a callback once all accesses got fetched */
        }
    }
    else
    {
        task_execute(runtime, device, task);
    }
}

/* main loop for the thread responsible the passed device */
static inline int
device_thread_main_loop(
    runtime_t * runtime,
    device_t * device,
    thread_t * thread
) {
    assert(thread == thread_t::get_tls());

# if XKRT_SUPPORT_JULIA
    // should not be needed when entering Julia code via `cfunction` or
    // `@ccallable` entry point
    // jl_adopt_thread();
# endif

    driver_t * driver = runtime->driver_get(device->driver_type);
    assert(driver);

    task_t * task = NULL;
    bool ready    = false;
    bool pending  = false;

    // test whether the thread should be put to sleep, all three conditions must be met:
    //  - the device is running
    //  - there is no ready tasks
    //  - there is no pending commands
    auto test = [&] (void)
    {
        // the device must stop
        if (device->state != XKRT_DEVICE_STATE_COMMIT)
            return false;

        // find a new task
        if (task == NULL)
            task = thread->worksteal();

        /// find commands pending or ready
        device->offloader_queues_are_empty(thread->tid, XKRT_QUEUE_TYPE_ALL, &ready, &pending);

        // is there is anything to progress, wake up
        if (task || ready || pending)
            return false;

        // else, sleep
        return true;
    };

    while (device->state == XKRT_DEVICE_STATE_COMMIT)
    {
        // pause the thread as long as the test returns 'true'
        if (runtime->conf.enable_busy_polling)
            test();
        else
            thread->pause(test);

        // if the runtime must stop, break
        if (device->state != XKRT_DEVICE_STATE_COMMIT)
        {
            assert(device->state == XKRT_DEVICE_STATE_STOP);
            break ;
        }

        // if there is a task ready, launch it
        if (task)
            __device_prepare_task(runtime, device, task);

        // if there are commands ready
        if (ready)
        {
            // launch them, and retrieve the number of newly pending commands
            int newly_pending = driver->device_offloader_launch(device, thread->tid);

            // if there is newly pending commands. ensure the pending flag is set
            if (newly_pending > 0)
                pending = true;
        }

        // if there are pending commands, progress them
        if (pending)
        {
            // no more tasks to launch
            // pause the thread until some progress has been made
            if (task == NULL && runtime->conf.enable_progress_thread_pause)
            {
                driver->device_offloader_wait_random_command(device, thread->tid);
            }
            // some task was ready, so maybe there is more.
            // Just poll events a bit (potentially unlocking more tasks)
            // and do another trip
            else
            {
                driver->device_offloader_progress(device, thread->tid);
            }
        }

        // task had been launched
        task = NULL;
    }


# if XKRT_SUPPORT_JULIA
    // TODO: jl_abandon_thread();
    // jl_detach_thread();
# endif

    return EINTR;
}

///////////
//  MAIN //
///////////

/* Main entry thread created per device */
void *
device_thread_main(
    runtime_t * runtime,
    team_t * team,
    thread_t * thread
) {
    // unpack args
    device_team_args_t * args = (device_team_args_t *) team->desc.args;
    assert(args);

    // unpack args runtime
    driver_t * driver                   = args->driver;
    device_driver_id_t device_driver_id = args->device_driver_id;
    device_unique_id_t device_unique_id = args->device_unique_id;

    // register the device thread
    thread->device_unique_id = device_unique_id;

    // get device
    device_t * device = runtime->device_get(device_unique_id);
    assert(device);

    // print thread
    unsigned int cpu, node;
    getcpu(&cpu, &node);
    LOGGER_INFO("Starting thread for %s device (device_driver_id=%d, device_unique_id=%d) on cpu %d of node %d",
            driver->f_get_name(), device_driver_id, device->unique_id, cpu, node);

    // initialize offloader thread to initialize queues
    driver->device_offloader_init_thread(device, thread->tid);

    // wait for all threads of that device to be initialized
    pthread_barrier_wait(&args->barrier);

    // wait for all devices of that driver to be initialized
    if (thread->tid == 0)
        pthread_barrier_wait(&driver->barrier);

    // wait for all drivers to be initialized
    if (thread->tid == 0 && device_driver_id == 0)
        pthread_barrier_wait(&runtime->drivers.barrier);

    /* infinite loop with the device context */
    int err = device_thread_main_loop(runtime, device, thread);
    assert((err==0) || (err==EINTR));

    // delete queues
    if (driver->f_command_queue_delete)
        for (uint8_t j = 0 ; j < XKRT_QUEUE_TYPE_ALL ; ++j)
            for (int k = 0 ; k < device->count[j] ; ++k)
                driver->f_command_queue_delete(device->queues[thread->tid][j][k]);

    return NULL;
}

void
runtime_t::task_thread_enqueue(
    thread_t * thread,
    task_t * task
) {
    thread->deque.push(task);

    // TODO: this is quite ugly, but the thread may be sleeping in three places:
    //  - within its condition
    //  - within a team barrier (thus, the broadcast)
    //  - within parallel for
    thread->wakeup();
    if (thread->team)
        pthread_cond_signal(&thread->team->priv.barrier.cond);
}

void
runtime_t::task_team_enqueue(
    team_t * team,
    task_t * task
) {
    // get a random thread in the team
    int nthreads = team->get_nthreads();
    thread_t * tls = thread_t::get_tls();
    assert(tls);
    int start = tls->rng() % nthreads;

    // assign it the task
    thread_t * thread = team->get_thread(start);
    return this->task_thread_enqueue(thread, task);
}

static inline void
__task_detachable_decr(void * args[XKRT_CALLBACK_ARGS_MAX])
{
    runtime_t * runtime = (runtime_t *) args[0];
    assert(runtime);

    task_t * task = (task_t *) args[1];
    assert(task);

    runtime->task_detachable_decr(task);
}

int
runtime_t::task_emit_command(
    const device_unique_id_t device_unique_id,
    const command_queue_type_t qtype,
    const ocg::command_type_t ctype,
    const command_flag_t flags,
    const std::function<void(command_t *)> builder
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task_t * task = thread->current_task;
    assert(task);
    assert(task->flags & TASK_FLAG_DETACHABLE);

    device_t * device = this->device_get(device_unique_id);
    assert(device);

    /* create a new command and retrieve its offload queue */
    thread_t * device_thread;
    command_queue_t * queue;
    command_t * cmd;
    device->offloader_queue_command_new(
        qtype,
        ctype,
        flags,
        &device_thread,
        &queue,
        &cmd
    );
    assert(device_thread);
    assert(queue);
    assert(cmd);

    /* build command */
    builder(cmd);

    /* increment detach counter and push callback to decrement */
    callback_t callback;
    callback.func    = __task_detachable_decr;
    callback.args[0] = this;
    callback.args[1] = task;
    assert(XKRT_CALLBACK_ARGS_MAX >= 2);

    this->task_detachable_incr(task);
    cmd->completion_callback_push(callback);

    device->offloader_queue_command_commit(device_thread, queue, cmd);
    return 0;
}

template <bool synchronous>
void
runtime_t::task_prog_launch(
    device_t * device,
    task_t * task,
    prog_launcher_t launcher
) {
    /* increase detach counter if asynchronous */
    if constexpr (synchronous == false)
    {
        assert(task->flags & TASK_FLAG_DETACHABLE);
        this->task_detachable_incr(task);

        /* the task may complete in the callback on kernel completion */
        callback_t callback;
        callback.func    = __task_detachable_decr;
        callback.args[0] = this;
        callback.args[1] = task;
        assert(XKRT_CALLBACK_ARGS_MAX >= 2);

        /* submit kernel launch command */
        device->offloader_queue_command_submit_kernel<COMMAND_FLAG_NONE>(
            this,
            task,
            launcher,
            callback
        );
    }
    /* else if synchronous, no callback or detach counter */
    else
    {
        static_assert(synchronous == true);
        device->offloader_queue_command_submit_kernel<COMMAND_FLAG_SYNCHRONOUS>(this, task, launcher);
    }
}
template void runtime_t::task_prog_launch<true>(device_t * device, task_t * task, prog_launcher_t launcher);
template void runtime_t::task_prog_launch<false>(device_t * device, task_t * task, prog_launcher_t launcher);

XKRT_NAMESPACE_END
