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
    bool allocated;

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
    device_unique_id_t unique_id;

    /* the device state */
    std::atomic<device_state_t> state;

    /* affinity[i] - j-th bit is set to '1' if this device has an affinity 'i'
     * with 'j' (the lowest affinity, the better perf) */
    device_unique_id_bitfield_t * affinity;

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
    command_queue_t ** queues[XKRT_MAX_THREADS_PER_DEVICE][XKRT_QUEUE_TYPE_ALL];

    /* progress pending commands in every queues */
    int offloader_wait_random_command(int tid);

    /* set 'ready' and 'pending' to false whether there is ready/pending
     * commands in the queues of the given type */
    void offloader_queues_are_empty(int tid, const command_queue_type_t qtype, bool * ready, bool * pending) const;

    /* get next queue to use for submitting a command for the given type */
    void offloader_queue_next(
        const command_queue_type_t type,
        thread_t ** pthread,        /* OUT */
        command_queue_t ** pqueue           /* OUT */
    );

    /* submit a command to the device */
    int command_submit(command_t * command);

    /* create a new command */
    void offloader_queue_command_new(
        const command_queue_type_t qtype,   /* IN  */
        const ocg::command_type_t ctype,         /* IN  */
        const command_flag_t flags,         /* IN  */
        thread_t ** pthread,                /* OUT */
        command_queue_t ** pqueue,          /* OUT */
        command_t ** pcmd                   /* OUT */
    );

    /* commit a command previously returned with
     * "offloader_queue_command_new" */
    void offloader_queue_command_commit(
        thread_t * thread,
        command_queue_t * queue,
        command_t * cmd
    );

    # pragma message(TODO "Remove all these routines to use a generic runtime API")

    /* submit a file I/O command */
    template <ocg::command_type_t T>
    command_t * offloader_queue_command_submit_file(
        int    fd,
        void * buffer,
        size_t size,
        size_t offset,
        const std::optional<callback_t> & callback = std::nullopt
    ) {
        static_assert(
            T == ocg::COMMAND_TYPE_FD_READ ||
            T == ocg::COMMAND_TYPE_FD_WRITE
        );

        /* create a new command and retrieve its offload queue */
        constexpr command_queue_type_t   qtype = (T == ocg::COMMAND_TYPE_FD_READ) ? XKRT_QUEUE_TYPE_FD_READ : XKRT_QUEUE_TYPE_FD_WRITE;
        constexpr ocg::command_type_t ctype = T;
        constexpr command_flag_t flags = COMMAND_FLAG_NONE;

        thread_t * thread;
        command_queue_t * queue;
        command_t * cmd;
        this->offloader_queue_command_new(qtype, ctype, flags, &thread, &queue, &cmd);
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
            cmd->completion_callback_push(*callback);
        this->offloader_queue_command_commit(thread, queue, cmd);

        return cmd;
    }

    template <command_flag_t flags>
    command_t * offloader_queue_command_submit_kernel(
        void * runtime,
        task_t * task,
        prog_launcher_t launch,
        const std::optional<callback_t> & callback = std::nullopt
    ) {
        /* create a new command and retrieve its offload queue */
        thread_t * thread;
        command_queue_t * queue;
        command_t * cmd;
        this->offloader_queue_command_new(
            XKRT_QUEUE_TYPE_KERN,
            ocg::COMMAND_TYPE_PROG_LAUNCHER,
            flags,
            &thread,
            &queue,
            &cmd
        );
        assert(thread);
        assert(queue);
        assert(cmd);

        /* create a new kernel command */
        cmd->prog_launcher.launch  = (void (*)()) launch;
        cmd->prog_launcher.runtime = runtime;
        cmd->prog_launcher.device  = this;
        cmd->prog_launcher.task    = task;
        if (callback)
            cmd->completion_callback_push(*callback);

        this->offloader_queue_command_commit(thread, queue, cmd);

        return cmd;
    }

    /* copy */
    template <typename HOST_VIEW_T, typename DEVICE_VIEW_T>
    command_t *
    offloader_queue_command_submit_copy(
        const HOST_VIEW_T               & host_view,
        const device_unique_id_t          dst_device_unique_id,
        const DEVICE_VIEW_T             & dst_device_view,
        const device_unique_id_t          src_device_unique_id,
        const DEVICE_VIEW_T             & src_device_view,
        const std::optional<callback_t> & callback = std::nullopt
    ) {
        assert(this->unique_id == dst_device_unique_id || this->unique_id == src_device_unique_id);

        /* find the command type */
        ocg::command_type_t ctype;
        const int src_is_host = (src_device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID) ? 1 : 0;
        const int dst_is_host = (dst_device_unique_id == XKRT_HOST_DEVICE_UNIQUE_ID) ? 1 : 0;

        /* assertions */
        # define IS_1D (std::is_same<HOST_VIEW_T, size_t>()        && std::is_same<DEVICE_VIEW_T, uintptr_t>())
        # define IS_2D (std::is_same<HOST_VIEW_T, memory_view_t>() && std::is_same<DEVICE_VIEW_T, memory_replica_view_t>())
        static_assert(IS_1D || IS_2D);
        if constexpr(IS_1D) {
            assert(host_view);
            assert(dst_device_view);
            assert(src_device_view);
            ctype = ( src_is_host &&  dst_is_host) ? ocg::COMMAND_TYPE_COPY_H2H_1D :
                    ( src_is_host && !dst_is_host) ? ocg::COMMAND_TYPE_COPY_H2D_1D :
                    (!src_is_host &&  dst_is_host) ? ocg::COMMAND_TYPE_COPY_D2H_1D :
                    (!src_is_host && !dst_is_host) ? ocg::COMMAND_TYPE_COPY_D2D_1D :
                    ocg::COMMAND_TYPE_MAX;
        } else if constexpr(IS_2D) {
            assert(host_view.m);
            assert(host_view.n);
            assert(host_view.sizeof_type);

            assert(dst_device_view.addr);
            assert(dst_device_view.ld);

            assert(src_device_view.addr);
            assert(src_device_view.ld);

            ctype = ( src_is_host &&  dst_is_host) ? ocg::COMMAND_TYPE_COPY_H2H_2D :
                    ( src_is_host && !dst_is_host) ? ocg::COMMAND_TYPE_COPY_H2D_2D :
                    (!src_is_host &&  dst_is_host) ? ocg::COMMAND_TYPE_COPY_D2H_2D :
                    (!src_is_host && !dst_is_host) ? ocg::COMMAND_TYPE_COPY_D2D_2D :
                    ocg::COMMAND_TYPE_MAX;
        } else {
            LOGGER_FATAL("Wrong parameters");
        }

        /* find the type of queue to use */
        command_queue_type_t qtype;
        switch(ctype)
        {
            case (ocg::COMMAND_TYPE_COPY_H2H_1D):
            case (ocg::COMMAND_TYPE_COPY_H2D_1D):
            case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            {
               qtype = XKRT_QUEUE_TYPE_H2D;
               break ;
            }

            case (ocg::COMMAND_TYPE_COPY_D2H_1D):
            case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            {
                qtype = XKRT_QUEUE_TYPE_D2H;
                break ;
            }

            case (ocg::COMMAND_TYPE_COPY_D2D_1D):
            case (ocg::COMMAND_TYPE_COPY_D2D_2D):
            {
                qtype = (src_device_unique_id == dst_device_unique_id) ? XKRT_QUEUE_TYPE_D2D : XKRT_QUEUE_TYPE_P2P;
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
        command_queue_t * queue;
        command_t * cmd;
        constexpr command_flag_t flags = COMMAND_FLAG_NONE;
        this->offloader_queue_command_new(qtype, ctype, flags, &thread, &queue, &cmd);
        assert(thread);
        assert(queue);
        assert(cmd);

        /* create a new copy command */
        if constexpr (IS_1D)
        {
            cmd->copy_1D.src_device_unique_id = src_device_unique_id;
            cmd->copy_1D.dst_device_unique_id = dst_device_unique_id;
            cmd->copy_1D.src_device_addr      = src_device_view;
            cmd->copy_1D.dst_device_addr      = dst_device_view;
            cmd->copy_1D.size                 = host_view;
        }
        else if constexpr (IS_2D)
        {
            cmd->copy_2D.src_device_unique_id = src_device_unique_id;
            cmd->copy_2D.dst_device_unique_id = dst_device_unique_id;
            cmd->copy_2D.src_addr             = src_device_view.addr;
            cmd->copy_2D.src_ld               = src_device_view.ld;
            cmd->copy_2D.dst_addr             = dst_device_view.addr;
            cmd->copy_2D.dst_ld               = dst_device_view.ld;
            cmd->copy_2D.m                    = host_view.m;
            cmd->copy_2D.n                    = host_view.n;
            cmd->copy_2D.sizeof_type          = host_view.sizeof_type;
        }

        if (callback)
            cmd->completion_callback_push(*callback);
        this->offloader_queue_command_commit(thread, queue, cmd);

        # undef IS_1D
        # undef IS_2D

        return cmd;
    }

}               device_t;

XKRT_NAMESPACE_END

#endif /* __XKRT_DEVICE_HPP__ */
