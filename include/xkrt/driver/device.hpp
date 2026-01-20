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

#ifndef __XKRT_DEVICE_HPP__
# define __XKRT_DEVICE_HPP__

# include <stdint.h>    /* uint64_t */

# include <xkrt/conf/conf.h>
# include <xkrt/driver/driver-type.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/todo.h>
# include <xkrt/memory/area.h>
# include <xkrt/memory/cache-line-size.hpp>
# include <xkrt/stats/stats.h>
# include <xkrt/support.h>
# include <xkrt/sync/mutex.h>
# include <xkrt/task/task.hpp>

# include <optional>

XKRT_NAMESPACE_BEGIN

typedef enum    device_state_t : uint8_t
{
    XKRT_DEVICE_STATE_DEALLOCATED = 0,
    XKRT_DEVICE_STATE_CREATE      = 1,
    XKRT_DEVICE_STATE_INIT        = 2,
    XKRT_DEVICE_STATE_COMMIT      = 3,
    XKRT_DEVICE_STATE_STOP        = 4,
    XKRT_DEVICE_STATE_STOPPED     = 5,
    XKRT_DEVICE_STATE_DESTROYED   = 6

}               device_state_t;

/* Memory info of a device */
typedef struct  device_memory_info_t
{
    ///////////////////////////////////////
    //  TO BE FILL BY THE DRIVER ON INIT //
    ///////////////////////////////////////

    /* memory capacity */
    size_t capacity;

    /* memory used */
    size_t used;

    /* memory name */
    char name[32];

    ////////////////////////////////
    //  TO BE FILL BY THE RUNTIME //
    ////////////////////////////////

    /* whether this area was already allocated+mapped to the device */
    int allocated;

    /* the area of that memory */
    area_t area;

}               device_memory_info_t;

/* A device virtualize a ressource with its one address space and
   a communication queue between host and the ressource */
typedef struct  device_t
{
    /////////////////
    //  ATTRIBUTES //
    /////////////////

    /* the conf */
    conf_device_t * conf;

    /* the driver type in [0..XKRT_DRIVER_TYPE_MAX[ */
    driver_type_t driver_type;

    /* driver device id in [0..ndevices_for_device] */
    device_driver_id_t driver_id;

    /* global device id in [0, XKRT_DEVICES_MAX[ - host is a virtual device of id 'XKRT_DEVICES_MAX' */
    device_global_id_t global_id;

    /* the device state */
    std::atomic<device_state_t> state;

    /* affinity[i] - j-th bit is set to '1' if this device has an affinity 'i'
     * with 'j' (the lowest affinity, the better perf) */
    device_global_id_bitfield_t * affinity;

    /* the device team */
    team_t * team;

    ///////////
    // STATS //
    ///////////

    # if XKRT_SUPPORT_STATS
    struct {
        struct {
            stats_int_t freed;
            struct {
                stats_int_t total;
                stats_int_t currently;
            } allocated;
        } memory;
    } stats;
    # endif /* XKRT_SUPPORT_STATS */

    //////////////////////
    // MEMORY MANAGMENT //
    //////////////////////

    /* memory areas of that device - sorted by performance */
    device_memory_info_t memories[XKRT_DEVICE_MEMORIES_MAX];
    int nmemories;

    /* allocate memory on a specific area */
    area_chunk_t * memory_allocate_on(const size_t size, int area_idx);

    /* allocate memory */
    area_chunk_t * memory_allocate(const size_t size);

    /* deallocate the given chunk */
    void memory_deallocate_on(area_chunk_t * chunk, int area_idx);

    /* deallocate the given chunk */
    void memory_deallocate(area_chunk_t * chunk);

    /* free all memory of every area of that device, resetting their state to chunk0 */
    void memory_reset(void);

    /* free all memory of the given area of that device, resetting their state to chunk0 */
    void memory_reset_on(int area_idx);

    /* set chunk0 of an area */
    void memory_set_chunk0(uintptr_t device_ptr, size_t size, int area_idx);

    ///////////////////////
    // QUEUE MANAGEMENT //
    ///////////////////////

    /* total number of queue (sum of count[:]) */
    int nqueues_per_thread;

    /* number of queue per type */
    int count[XKRT_QUEUE_TYPE_ALL];

    /* next thread to use for offloading a command */
    std::atomic<int> next_thread;

    /* next queue to use for the given thread and type */
    std::atomic<int> next_queue[XKRT_MAX_THREADS_PER_DEVICE][XKRT_QUEUE_TYPE_ALL];

    /* basic queue */
    queue_t ** queues[XKRT_MAX_THREADS_PER_DEVICE][XKRT_QUEUE_TYPE_ALL];

    /* initialize the offloader (must be called once before any thread called the 'thread' version) */
    void offloader_init(
        int (*f_queue_suggest)(device_driver_id_t device_driver_id, queue_type_t type)
    );

    /* initialize a thread of the offloader */
    void offloader_init_thread(
        int tid,
        queue_t * (*f_queue_create)(device_t * device, queue_type_t type, queue_command_list_counter_t capacity)
    );

    /* launch ready commands in every queues */
    int offloader_launch(int tid);

    /* progress pending commands in every queues */
    int offloader_progress(int tid);

    /* progress pending commands in every queues */
    int offloader_wait_random_command(int tid);

    /* set 'ready' and 'pending' to false whether there is ready/pending
     * commands in the queues of the given type */
    void offloader_queues_are_empty(int tid, const queue_type_t qtype, bool * ready, bool * pending) const;

    /* get next queue to use for submitting a command for the given type */
    void offloader_queue_next(
        const queue_type_t type,
        thread_t ** pthread,        /* OUT */
        queue_t ** pqueue           /* OUT */
    );

    /* launch ready commands dispatching them in queues of the given type */
    int offloader_queue_commands_launch(int tid, const queue_type_t qtype);

    /* progress pending commands in queues of the given type of the given thread.
     * If blocking is true, also waits for the completion of pending commands */
    template <bool blocking>
    int
    offloader_queue_commands_progress(
        int tid,
        const queue_type_t qtype
    ) {
        int err = 0;
        unsigned int bgn = (qtype == XKRT_QUEUE_TYPE_ALL) ?               0 : qtype;
        unsigned int end = (qtype == XKRT_QUEUE_TYPE_ALL) ? XKRT_QUEUE_TYPE_ALL : qtype + 1;
        for (unsigned int s = bgn ; s < end ; ++s)
        {
            for (int i = 0 ; i < this->count[s] ; ++i)
            {
                queue_t * queue = this->queues[tid][s][i];
                assert(queue);

                if (queue->pending.is_empty())
                    continue ;

                queue_command_list_counter_t n;
                do {
                    queue->lock();
                    if (blocking)
                    {
                        queue->wait_pending_commands();
                        err = 0;
                    }
                    else
                        err = queue->progress_pending_commands();
                    queue->unlock();
                    n = queue->pending.size();
                    assert(n < queue->pending.capacity);
                } while (n > this->conf->offloader.queues[s].concurrency);
                assert(err == 0 || err == EINPROGRESS);
            }
        }
        return 0;
    }

    /* create a new command and lock the queue */
    template<bool synchronous>
    void offloader_queue_command_new(
        const queue_type_t qtype,       /* IN  */
              thread_t ** pthread,      /* OUT */
              queue_t ** pqueue,        /* OUT */
        const command_type_t ctype,     /* IN  */
              command_t ** pcmd         /* OUT */
    ) {
        assert(pqueue);
        assert(pcmd);

        /* retrieve native queue */
        this->offloader_queue_next(qtype, pthread, pqueue);
        assert(*pthread);
        assert(*pqueue);
        assert((*pqueue)->type == qtype);

        /* allocate the command */
        do {
            (*pqueue)->lock();
            (*pcmd) = (*pqueue)->command_new<synchronous>(ctype);
            if (*pcmd)
                break ;
            (*pqueue)->unlock();

            LOGGER_FATAL("Stream is full, increase 'XKRT_OFFLOADER_CAPACITY' or implement support for full-queue management yourself :-) (sorry)");

        } while (1);

        /* queue is locked, will be unlocked in the commit */
    }

    /* commit a command previously returned with
     * "offloader_queue_command_new" and unlock the queue */
    void offloader_queue_command_commit(
        thread_t * thread,
        queue_t * queue,
        command_t * cmd
    );

    /* submit a file I/O command */
    template <command_type_t T>
    command_t * offloader_queue_command_submit_file(
        int    fd,
        void * buffer,
        size_t size,
        size_t offset,
        const std::optional<callback_t> & callback = std::nullopt
    ) {
        static_assert(
            T == COMMAND_TYPE_FD_READ ||
            T == COMMAND_TYPE_FD_WRITE
        );

        /* create a new command and retrieve its offload queue */
        constexpr queue_type_t   qtype = (T == COMMAND_TYPE_FD_READ) ? XKRT_QUEUE_TYPE_FD_READ : XKRT_QUEUE_TYPE_FD_WRITE;
        constexpr command_type_t ctype = T;

        thread_t * thread;
        queue_t * queue;
        command_t * cmd;
        constexpr bool synchronous = false;
        this->offloader_queue_command_new<synchronous>(qtype, &thread, &queue, ctype, &cmd);
        assert(thread);
        assert(queue);
        assert(cmd);

        /* create a new file i/o command */
        cmd->file.fd = fd;
        cmd->file.buffer = buffer;
        cmd->file.size = size;
        cmd->file.offset = offset;

        /* submit cmd */
        if (callback)
            cmd->push_callback(*callback);
        this->offloader_queue_command_commit(thread, queue, cmd);

        return cmd;
    }

    template <bool synchronous>
    command_t * offloader_queue_command_submit_kernel(
        void * runtime,
        task_t * task,
        kernel_launcher_t launch,
        const std::optional<callback_t> & callback = std::nullopt
    ) {
        /* create a new command and retrieve its offload queue */
        thread_t * thread;
        queue_t * queue;
        command_t * cmd;
        this->offloader_queue_command_new<synchronous>(
            XKRT_QUEUE_TYPE_KERN,   /* IN */
            &thread,                /* OUT */
            &queue,                 /* OUT */
            COMMAND_TYPE_KERN,      /* IN */
            &cmd                    /* OUT */
        );
        assert(thread);
        assert(queue);
        assert(cmd);
        assert(queue->is_locked());

        /* create a new kernel command */
        cmd->kern.launch  = (void (*)()) launch;
        cmd->kern.runtime = runtime;
        cmd->kern.device  = this;
        cmd->kern.task    = task;
        if (callback)
            cmd->push_callback(*callback);

        this->offloader_queue_command_commit(thread, queue, cmd);

        return cmd;
    };

    /* copy */
    template <typename HOST_VIEW_T, typename DEVICE_VIEW_T>
    command_t *
    offloader_queue_command_submit_copy(
        const HOST_VIEW_T               & host_view,
        const device_global_id_t          dst_device_global_id,
        const DEVICE_VIEW_T             & dst_device_view,
        const device_global_id_t          src_device_global_id,
        const DEVICE_VIEW_T             & src_device_view,
        const std::optional<callback_t> & callback = std::nullopt
    ) {
        assert(this->global_id == dst_device_global_id || this->global_id == src_device_global_id);

        /* find the command type */
        command_type_t ctype;
        const int src_is_host = (src_device_global_id == HOST_DEVICE_GLOBAL_ID) ? 1 : 0;
        const int dst_is_host = (dst_device_global_id == HOST_DEVICE_GLOBAL_ID) ? 1 : 0;

        /* assertions */
        # define IS_1D (std::is_same<HOST_VIEW_T, size_t>()        && std::is_same<DEVICE_VIEW_T, uintptr_t>())
        # define IS_2D (std::is_same<HOST_VIEW_T, memory_view_t>() && std::is_same<DEVICE_VIEW_T, memory_replica_view_t>())
        static_assert(IS_1D || IS_2D);
        if constexpr(IS_1D) {
            assert(host_view);
            assert(dst_device_view);
            assert(src_device_view);
            ctype = ( src_is_host &&  dst_is_host) ? COMMAND_TYPE_COPY_H2H_1D :
                    ( src_is_host && !dst_is_host) ? COMMAND_TYPE_COPY_H2D_1D :
                    (!src_is_host &&  dst_is_host) ? COMMAND_TYPE_COPY_D2H_1D :
                    (!src_is_host && !dst_is_host) ? COMMAND_TYPE_COPY_D2D_1D :
                    COMMAND_TYPE_MAX;
        } else if constexpr(IS_2D) {
            assert(host_view.m);
            assert(host_view.n);
            assert(host_view.sizeof_type);

            assert(dst_device_view.addr);
            assert(dst_device_view.ld);

            assert(src_device_view.addr);
            assert(src_device_view.ld);

            ctype = ( src_is_host &&  dst_is_host) ? COMMAND_TYPE_COPY_H2H_2D :
                    ( src_is_host && !dst_is_host) ? COMMAND_TYPE_COPY_H2D_2D :
                    (!src_is_host &&  dst_is_host) ? COMMAND_TYPE_COPY_D2H_2D :
                    (!src_is_host && !dst_is_host) ? COMMAND_TYPE_COPY_D2D_2D :
                    COMMAND_TYPE_MAX;
        } else {
            LOGGER_FATAL("Wrong parameters");
        }

        /* find the type of queue to use */
        queue_type_t qtype;
        switch(ctype)
        {
            case (COMMAND_TYPE_COPY_H2H_1D):
            case (COMMAND_TYPE_COPY_H2D_1D):
            case (COMMAND_TYPE_COPY_H2H_2D):
            case (COMMAND_TYPE_COPY_H2D_2D):
            {
               qtype = XKRT_QUEUE_TYPE_H2D;
               break ;
            }

            case (COMMAND_TYPE_COPY_D2H_1D):
            case (COMMAND_TYPE_COPY_D2H_2D):
            {
                qtype = XKRT_QUEUE_TYPE_D2H;
                break ;
            }

            case (COMMAND_TYPE_COPY_D2D_1D):
            case (COMMAND_TYPE_COPY_D2D_2D):
            {
                qtype = (src_device_global_id == dst_device_global_id) ? XKRT_QUEUE_TYPE_D2D : XKRT_QUEUE_TYPE_P2P;
                break ;
            }

            default:
            {
                LOGGER_FATAL("Impossible occured");
                break ;
            }
        }

        /* create a new command and retrieve its offload queue */
        thread_t * thread;
        queue_t * queue;
        command_t * cmd;
        constexpr bool synchronous = false;
        this->offloader_queue_command_new<synchronous>(qtype, &thread, &queue, ctype, &cmd);
        assert(thread);
        assert(queue);
        assert(cmd);

        /* create a new copy command */
        if constexpr (IS_1D) {
            cmd->copy_1D.size = host_view;
            cmd->copy_1D.dst_device_addr  = dst_device_view;
            cmd->copy_1D.src_device_addr  = src_device_view;
            XKRT_STATS_INCR(queue->stats.transfered, cmd->copy_1D.size);
        } else if constexpr (IS_2D) {
            cmd->copy_2D.m                = host_view.m;
            cmd->copy_2D.n                = host_view.n;
            cmd->copy_2D.sizeof_type      = host_view.sizeof_type;
            cmd->copy_2D.dst_device_view  = dst_device_view;
            cmd->copy_2D.src_device_view  = src_device_view;
            XKRT_STATS_INCR(queue->stats.transfered, host_view.m * host_view.n * host_view.sizeof_type);
        }

        if (callback)
            cmd->push_callback(*callback);
        this->offloader_queue_command_commit(thread, queue, cmd);

        # undef IS_1D
        # undef IS_2D

        return cmd;
    }

}               device_t;

XKRT_NAMESPACE_END

#endif /* __XKRT_DEVICE_HPP__ */
