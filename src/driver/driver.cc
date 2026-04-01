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

# include <xkrt/support.h>
# include <xkrt/runtime.h>
# include <xkrt/internals.h>
# include <xkrt/driver/driver.h>
# include <xkrt/logger/logger.h>
# include <xkrt/utils/min-max.h>
# include <xkrt/sync/spinlock.h>
# include <xkrt/thread/thread.h>

# include <cassert>
# include <cstring>
# include <cerrno>
# include <climits>

XKRT_NAMESPACE_BEGIN;

static void
bits_to_str(char * buffer, unsigned char * mem, size_t nbytes)
{
    buffer[8*nbytes] = 0;
    size_t k = 8*nbytes - 1;
    for (int i = (int)nbytes - 1 ; i >= 0 ; --i)
        for (int j = 0 ; j < 8 ; ++j)
            buffer[k--] = (mem[i] & (1 << j)) ? '1' : '0';
}

/* entry point for threads in the driver's team */
static void *
driver_thread_main(
    runtime_t * runtime,
    team_t * team,
    thread_t * thread
) {
    // driver type
    driver_t * driver = (driver_t *) team->desc.args;
    assert(driver);

    // device driver id is the thread tid
    device_driver_id_t device_driver_id = (device_driver_id_t) thread->tid;
    assert(device_driver_id >= 0);
    assert(device_driver_id < driver->devices.n);

    // device global id
    device_unique_id_t device_unique_id = driver->devices.unique_ids[device_driver_id];

    ///////////////////////
    // create the device //
    ///////////////////////

    assert(driver->f_device_create);

    // create the device
    device_t * device = driver->f_device_create(driver, device_driver_id);
    if (device == NULL)
        LOGGER_FATAL("Could not create a device");

    // initialize device attributes
    device->state       = XKRT_DEVICE_STATE_CREATE;
    device->driver_type = driver->type;
    device->driver_id   = device_driver_id;
    device->conf        = &(runtime->conf.device);
    device->unique_id   = device_unique_id;

    // register device to the global list
    runtime->drivers.devices.list[device_unique_id] = device;

    // register device to the driver list
    driver->devices.list[device_driver_id] = device;

    // init device by the driver
    driver->f_device_init(device->driver_id);

    char buffer[512];
    driver->f_device_info(device_driver_id, buffer, sizeof(buffer));
    LOGGER_INFO("  global id = %2u | %s", device_unique_id, buffer);

    /* get total memory and allocate chunk0 */
    if (driver->f_memory_device_info)
    {
        driver->f_memory_device_info(device->driver_id, device->memories, &device->nmemories);
        assert(device->nmemories > 0);
        for (int i = 0 ; i < device->nmemories ; ++i)
        {
            device_memory_info_t * info = device->memories + i;
            LOGGER_INFO("Found memory `%s` of capacity %zuGB", info->name, info->capacity/(size_t)1e9);
            info->allocated = 0;
            XKRT_MUTEX_INIT(info->area.lock);
        }
    }

    // wait for all devices of that driver to be in the 'init' state
    assert(device->state == XKRT_DEVICE_STATE_CREATE);
    device->state = XKRT_DEVICE_STATE_INIT;
    pthread_barrier_wait(&driver->barrier);

    // commit
    assert(driver->f_device_commit);
    device_unique_id_bitfield_t * affinity = &(runtime->router.affinity[device->unique_id][0]);
    memset(affinity, 0, sizeof(runtime->router.affinity[device->unique_id]));
    int err = driver->f_device_commit(device->driver_id, affinity);
    if (err)
        LOGGER_FATAL("Commit fail device %d of driver %s", device->driver_id, driver->f_get_name());
    assert(device->state == XKRT_DEVICE_STATE_INIT);
    device->state = XKRT_DEVICE_STATE_COMMIT;

    // can only have 1 host device, that is the device 0
    assert(driver->type != XKRT_DRIVER_TYPE_HOST || driver->devices.n == 1);

    // print affinity
    for (int i = 0 ; i < XKRT_DEVICES_PERF_RANK_MAX ; ++i)
    {
        device_unique_id_bitfield_t bf = affinity[i];
        constexpr int nbytes = sizeof(device_unique_id_bitfield_t);
        char buffer[8*nbytes + 1];
        bits_to_str(buffer, (unsigned char *) &bf, nbytes);
        LOGGER_DEBUG("Device `%2u` affinity mask for perf `%2u` is `%s`", device->unique_id, i, buffer);
    }

    // init offloader
    driver->device_offloader_init(device);

    // wait for all devices to be in the 'commit' state with the offloader init
    pthread_barrier_wait(&driver->barrier);

    //////////////////////////////////////////////////
    // Fork a team of worker thread for that device //
    //////////////////////////////////////////////////

    // number of thread per device
    conf_driver_t * driver_conf = runtime->conf.drivers.list + driver->type;
    int nthreads_per_device = driver_conf->nthreads_per_device;
    assert(nthreads_per_device > 0);

    // the device team args
    device_team_args_t args = {
        .driver = driver,
        .device_unique_id = device_unique_id,
        .device_driver_id = device_driver_id,
        .barrier = {}
    };

    // prepare a barrier, to synchronize threads of that device
    if (pthread_barrier_init(&args.barrier, NULL, nthreads_per_device))
        LOGGER_FATAL("Couldnt initialized pthread_barrier_t");

    // create the device team
    device->team = driver->devices.teams + device_driver_id;

    device->team->desc.args                = &args;
    device->team->desc.binding.flags       = XKRT_TEAM_BINDING_FLAG_NONE;
    device->team->desc.binding.mode        = XKRT_TEAM_BINDING_MODE_COMPACT;
    device->team->desc.binding.nplaces     = 1;
    device->team->desc.binding.places      = XKRT_TEAM_BINDING_PLACES_EXPLICIT;
    device->team->desc.binding.places_list = team->desc.binding.places_list + device_driver_id;
    device->team->desc.master_is_member    = true;
    device->team->desc.nthreads            = nthreads_per_device;
    device->team->desc.routine             = (team_routine_t) device_thread_main;

    runtime->team_create(device->team);     // return from the 'device team'
    runtime->team_join(device->team);

    /////////////////////
    // Teardown driver //
    /////////////////////

    // release memory
    if (driver->f_memory_device_deallocate)
    {
        for (int j = 0 ; j < device->nmemories ; ++j)
        {
            if (device->memories[j].allocated)
            {
                area_t * area = &(device->memories[j].area);
                driver->f_memory_device_deallocate(device->driver_id, (void *) area->chunk0.ptr, area->chunk0.size, j);
            }
        }
    }
    else
        LOGGER_WARN("Driver `%u` is missing `f_device_memory_deallocate`", driver->type);

    // delete device
    if (driver->f_device_destroy)
        driver->f_device_destroy(device->driver_id);
    else
        LOGGER_WARN("Driver `%u` is missing `f_device_destroy`", driver->type);

    return NULL; // return from the 'driver team'
}

/* initialize drivers and create 1 thread per gpu starting on the passed routine */
void
drivers_init(runtime_t * runtime)
{
    # pragma message(TODO "Dynamic driver loading not implemented (with dlopen). Only supporting built-in drivers")

    // PARAMETERS
    device_unique_id_t ndevices_requested  = runtime->conf.device.ngpus + 1; // host device + ngpus
    bool use_p2p = runtime->conf.device.use_p2p;
    assert(ndevices_requested < XKRT_DEVICES_MAX);

    // SET MEMBERS
    memset(runtime->drivers.list, 0, sizeof(runtime->drivers.list));
    memset(runtime->drivers.devices.list, 0, sizeof(runtime->drivers.devices.list));
    runtime->drivers.devices.n = 0;

    // LOAD DRIVERS
    driver_t * (*creators[XKRT_DRIVER_TYPE_MAX])(void);
    memset(creators, 0, sizeof(creators));

    extern driver_t * XKRT_DRIVER_TYPE_HOST_create_driver(void);
    creators[XKRT_DRIVER_TYPE_HOST] = XKRT_DRIVER_TYPE_HOST_create_driver;
    static_assert(XKRT_DRIVER_TYPE_HOST == 0);
    static_assert(XKRT_HOST_DEVICE_UNIQUE_ID == 0);

    char support[512];
    strcpy(support, "host");

# if XKRT_SUPPORT_CUDA
    extern driver_t * XKRT_DRIVER_TYPE_CU_create_driver(void);
    creators[XKRT_DRIVER_TYPE_CUDA] = XKRT_DRIVER_TYPE_CU_create_driver;
    strcat(support, ", cuda");
# endif /* XKRT_SUPPORT_CUDA */

# if XKRT_SUPPORT_ZE
    extern driver_t * XKRT_DRIVER_TYPE_ZE_create_driver(void);
    creators[XKRT_DRIVER_TYPE_ZE] = XKRT_DRIVER_TYPE_ZE_create_driver;
    strcat(support, ", ze");
# endif /* XKRT_SUPPORT_ZE */

# if XKRT_SUPPORT_CL
    extern driver_t * XKRT_DRIVER_TYPE_CL_create_driver(void);
    creators[XKRT_DRIVER_TYPE_CL] = XKRT_DRIVER_TYPE_CL_create_driver;
    strcat(support, ", opencl");
# endif /* XKRT_SUPPORT_CL */

# if XKRT_SUPPORT_HIP
    extern driver_t * XKRT_DRIVER_TYPE_HIP_create_driver(void);
    creators[XKRT_DRIVER_TYPE_HIP] = XKRT_DRIVER_TYPE_HIP_create_driver;
    strcat(support, ", hip");
# endif /* XKRT_SUPPORT_HIP */

# if XKRT_SUPPORT_SYCL
    extern driver_t * XKRT_DRIVER_TYPE_SYCL_create_driver(void);
    creators[XKRT_DRIVER_TYPE_SYCL] = XKRT_DRIVER_TYPE_SYCL_create_driver;
    strcat(support, ", sycl");
# endif /* XKRT_SUPPORT_SYCL */

    LOGGER_INFO("Built with support for `%s`", support);

    // TODO: currently sequentially initializing driver.
    // Maybe we should initialize them in parallel, but that'd break XKAAPI_NGPUS semantic

    ////////////////////////////////////////////////
    // First, figure-out how many drivers we have //
    ////////////////////////////////////////////////

    int ndrivers = 0;

    // for each driver
    for (uint8_t driver_type = 0 ; driver_type < XKRT_DRIVER_TYPE_MAX && runtime->drivers.devices.n < ndevices_requested ; ++driver_type)
    {
        // if the driver is enabled
        driver_t * (*creator)(void) = creators[driver_type];
        conf_driver_t * driver_conf = runtime->conf.drivers.list + driver_type;
        if (driver_conf->used && creator)
        {
            // instanciate it
            driver_t * driver = creator();
            runtime->drivers.list[driver_type] = driver;
            if (driver == NULL)
                continue ;
            driver->type = (driver_type_t) driver_type;

            // instanciate devices
            const char * driver_name = driver->f_get_name ? driver->f_get_name() : "(null)";
            LOGGER_INFO("Loading driver `%s`", driver_name);

            driver->devices.n = 0;
            if (driver->f_init == NULL || driver->f_init(ndevices_requested - runtime->drivers.devices.n, use_p2p))
            {
                LOGGER_WARN("Failed to load");
                continue ;
            }

            // number of devices for that driver
            assert(driver->f_get_ndevices_max);
            unsigned int ndevices_max = driver->f_get_ndevices_max();
            LOGGER_DEBUG("Driver has up to %u devices", ndevices_max);

            if (ndevices_max == 0)
                continue ;

            driver->devices.n = (device_driver_id_t) MIN(ndevices_requested - runtime->drivers.devices.n, ndevices_max);
            assert(driver->devices.n);

            ++ndrivers;

            ////////////////////////////////////////////////////////////////////
            // create a team with 1 thread per device                         //
            // each thread then forks a new team of threads on that device    //
            ////////////////////////////////////////////////////////////////////

            // generate places and global ids
            team_thread_place_t * places = (team_thread_place_t *) malloc(sizeof(team_thread_place_t) * driver->devices.n);
            assert(places);

            driver->devices.bitfield = 0;

            for (device_driver_id_t device_driver_id = 0; device_driver_id < driver->devices.n ; ++device_driver_id)
            {
                assert(driver->f_device_cpuset);
                int err = driver->f_device_cpuset(runtime->topology, places + device_driver_id, device_driver_id);
                if (err)
                {
                    LOGGER_WARN("Invalid cpuset returned for device %d - using default cpuset", device_driver_id);
                    CPU_ZERO(places + device_driver_id);
                    long nproc = sysconf(_SC_NPROCESSORS_CONF);
                    for (int i = 0; i < nproc; ++i)
                        CPU_SET(i, places + device_driver_id);
                }

                // set the device global id
                static_assert(XKRT_DRIVER_TYPE_HOST == 0);
                static_assert(XKRT_HOST_DEVICE_UNIQUE_ID == 0);
                const device_unique_id_t device_unique_id = runtime->drivers.devices.n++;
                driver->devices.unique_ids[device_driver_id] = device_unique_id;
                driver->devices.bitfield |= (1 << device_unique_id);
            }

            // create the driver team
            driver->team.desc.args                = driver;
            driver->team.desc.binding.flags       = XKRT_TEAM_BINDING_FLAG_NONE;
            driver->team.desc.binding.mode        = XKRT_TEAM_BINDING_MODE_COMPACT;
            driver->team.desc.binding.nplaces     = driver->devices.n;
            driver->team.desc.binding.places      = XKRT_TEAM_BINDING_PLACES_EXPLICIT;
            driver->team.desc.binding.places_list = places;
            driver->team.desc.master_is_member    = false;
            driver->team.desc.nthreads            = driver->devices.n;
            driver->team.desc.routine             = (team_routine_t) driver_thread_main;

            // prepare a barrier, to synchronize devices of that driver
            if (pthread_barrier_init(&driver->barrier, NULL, driver->devices.n))
                LOGGER_FATAL("Couldnt initialized pthread_barrier_t");
        }
        else
        {
            runtime->drivers.list[driver_type] = NULL;
        }
    }
    assert(runtime->drivers.devices.n <= ndevices_requested);

    // prepare a barrier, to synchronize drivers
    if (pthread_barrier_init(&runtime->drivers.barrier, NULL, ndrivers + 1))
        LOGGER_FATAL("Couldnt initialized pthread_barrier_t");

    ////////////////////////////////////////////////
    // Second, init device of each driver         //
    ////////////////////////////////////////////////

    // for each driver
    for (uint8_t driver_type = 0 ; driver_type < XKRT_DRIVER_TYPE_MAX ; ++driver_type)
    {
        // if the driver is enable, create a team of thread
        driver_t * driver = runtime->driver_get((driver_type_t) driver_type);
        if (driver && driver->devices.n)
            runtime->team_create(&driver->team);
    }

    // wait for all devices to be created
    pthread_barrier_wait(&runtime->drivers.barrier);

    // DEBUG OUTPUT
    if (runtime->drivers.devices.n == 0)
        LOGGER_WARN("No devices found :-(");
    else
        LOGGER_INFO("Found %d devices (with %d requested)", runtime->drivers.devices.n, ndevices_requested);
}

void
drivers_deinit(runtime_t * runtime)
{
    // notify each thread to stop
    for (device_unique_id_t device_unique_id = 0 ; device_unique_id < runtime->drivers.devices.n ; ++device_unique_id)
    {
        device_t * device = runtime->drivers.devices.list[device_unique_id];
        assert(device);
        device->state = XKRT_DEVICE_STATE_STOP;
        device->team->wakeup();
    }

    // finalize each driver
    for (uint8_t driver_type = 0 ; driver_type < XKRT_DRIVER_TYPE_MAX ; ++driver_type)
    {
        // join threads
        driver_t * driver = runtime->drivers.list[driver_type];
        if (driver && driver->devices.n)
        {
            runtime->team_join(&driver->team);
            free(driver->team.desc.binding.places_list);
        }

        // finalize driver
        if (driver)
        {
            if (driver->f_finalize)
                driver->f_finalize();
            else
                LOGGER_WARN("Driver `%u` is missing `f_finalize`", driver_type);
        }
    }
}

device_t *
driver_device_get(driver_t * driver, device_driver_id_t device_driver_id)
{
    assert(device_driver_id >= 0);
    assert(device_driver_id < driver->devices.n);
    return driver->devices.list[device_driver_id];
}

/////////////////////////////////
// DEVICE AND QUEUE MANAGEMENT //
/////////////////////////////////

void
driver_t::device_offloader_init(device_t * device)
{
    assert(device->driver_type == this->type);

    /* next queue to use (round robin) */
    device->next_thread = 0;
    memset(device->next_queue, 0, sizeof(device->next_queue));

    /* count total number of queue */
    device->nqueues_per_thread = 0;

    for (int qtype = 0 ; qtype < XKRT_QUEUE_TYPE_ALL ; ++qtype)
    {
        device->count[qtype] = (device->conf->offloader.queues[qtype].n >= 0) ? device->conf->offloader.queues[qtype].n : this->f_command_queue_suggest ? this->f_command_queue_suggest(device->driver_id, (command_queue_type_t) qtype) : 4;
        device->nqueues_per_thread += device->count[qtype];
    }
}

void
driver_t::device_offloader_init_thread(
    device_t * device,
    int tid
) {
    assert(device->driver_type == this->type);
    if (device->nqueues_per_thread == 0)
        return ;

    /* allocate queues array */
    assert(device->nqueues_per_thread);
    command_queue_t ** all_queues = (command_queue_t **) malloc(sizeof(command_queue_t *) * device->nqueues_per_thread);
    assert(all_queues);

    /* retrieve queue offset per type */
    uint16_t i = 0;
    for (int qtype = 0 ; qtype < XKRT_QUEUE_TYPE_ALL ; ++qtype)
    {
        device->queues[tid][qtype] = all_queues + i;
        for (int j = 0 ; j < device->count[qtype] ; ++j, ++i)
        {
            // create a new queue
            all_queues[i] = this->f_command_queue_create(device, static_cast<command_queue_type_t>(qtype), device->conf->offloader.capacity);
            if (all_queues[i] == NULL)
            {
                device->count[qtype] = j;
                break ;
            }
        }
    }
    assert(i <= device->nqueues_per_thread);
}

static inline int
driver_device_command_queue_launch_ready(
    driver_t * driver,
    device_t * device,
    command_queue_t * queue
) {
    assert(driver->type == device->driver_type);

    if (queue->ready.is_empty())
        return 0;

    int r = 0;

    /* for each ready command */
    SPINLOCK_LOCK(queue->spinlock);
    const xkrt_command_queue_list_counter_t p = queue->ready.iterate([&] (xkrt_command_queue_list_counter_t p) {

        /* if the pending queue is full, we cannot start more commands */
        if (queue->pending.is_full())
            return false;

        /* retrieve it */
        command_t * cmd = queue->ready.cmd + p;
        assert(cmd);

        LOGGER_DEBUG(
            "Decoding command `%s` on queue %p of type `%s` - p=%u, r=%u, w=%u",
            ocg::command_type_to_str(cmd->type),
            queue,
            command_queue_type_to_str(queue->type),
            p,
            queue->ready.pos.r,
            queue->ready.pos.w
        );

        /* launch command */
        switch (cmd->type)
        {
            /* Empty commands need not to execute anything */
            case (ocg::COMMAND_TYPE_EMPTY):
            {
                break ;
            }

            /* custom kernel launcher commands are launched by the command
             * itself, not the driver */
            case (ocg::COMMAND_TYPE_PROG_LAUNCHER):
            {
                ((prog_launcher_t) cmd->prog_launcher.launch)(
                    cmd->prog_launcher.runtime,
                    cmd->prog_launcher.device,
                    cmd->prog_launcher.task,
                    queue,
                    cmd,
                    p
                );
                break ;
            }

            case (ocg::COMMAND_TYPE_COPY_H2H_1D):
            case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            {
                LOGGER_FATAL("Not implemented");
                break ;
            }

            /******************************/
            /* launch commands via driver */
            /******************************/

            case (ocg::COMMAND_TYPE_BATCH):
            case (ocg::COMMAND_TYPE_PROG):
            case (ocg::COMMAND_TYPE_COPY_H2D_1D):
            case (ocg::COMMAND_TYPE_COPY_D2H_1D):
            case (ocg::COMMAND_TYPE_COPY_D2D_1D):
            case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            case (ocg::COMMAND_TYPE_COPY_D2D_2D):
            case (ocg::COMMAND_TYPE_FD_READ):
            case (ocg::COMMAND_TYPE_FD_WRITE):
            default:
            {
                int err = driver->f_command_queue_launch(device->driver_id, queue, cmd, p);
                switch (err)
                {
                    case (0):
                    case (EINPROGRESS):
                    {
                        break ;
                    }

                    case (ENOSYS):
                    {
                        LOGGER_FATAL("Command `%s` not implemented", ocg::command_type_to_str(cmd->type));
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
        if (cmd->flags & COMMAND_FLAG_SYNCHRONOUS)
        {
            // TODO: may lead to deadlock if reentrant
            queue->complete_command(p);
        }
        /* else, save to pending list */
        else
        {
            /* the pending queue must not be full at that point */
            assert(!queue->pending.is_full());
            const xkrt_command_queue_list_counter_t wp = queue->pending.pos.w;
            queue->pending.pos.w = (queue->pending.pos.w + 1) % queue->pending.capacity;

            memcpy(
                (void *) (queue->pending.cmd + wp),
                (void *) (queue->ready.cmd   + p),
                sizeof(command_t)
            );

            ++r;
        }

        /* continue */
        LOGGER_DEBUG("(loop) ready.is_empty() = %d, pending.is_empty() = %d", queue->ready.is_empty(), queue->pending.is_empty());
        return true;

    }); /* RING_ITERATE */
    SPINLOCK_UNLOCK(queue->spinlock);

    // this barrier ensures that the threads that owns the queue correctly
    // sees the 'ready' queue empty but not the 'pending' - else it would go
    // to sleep even though there is pending commands
    writemem_barrier();
    queue->ready.pos.r = p;

    LOGGER_DEBUG("ready.is_empty() = %d, pending.is_empty() = %d",
            queue->ready.is_empty(), queue->pending.is_empty());

    return r;
}

int
driver_t::device_offloader_launch(
    device_t * device,
    int tid,
    const command_queue_type_t qtype
) {
    int r = 0;

    unsigned int bgn = (qtype == XKRT_QUEUE_TYPE_ALL) ?                    0 : qtype;
    unsigned int end = (qtype == XKRT_QUEUE_TYPE_ALL) ? XKRT_QUEUE_TYPE_ALL : qtype + 1;
    for (unsigned int s = bgn ; s < end ; ++s)
    {
        for (int i = 0 ; i < device->count[s] ; ++i)
        {
            command_queue_t * queue = device->queues[tid][s][i];
            assert(queue);

            r += driver_device_command_queue_launch_ready(this, device, queue);
        }
    }

    return r;
}

template <bool blocking>
int
driver_t::device_offloader_progress(
    device_t * device,
    int tid,
    const command_queue_type_t qtype
) {
    int err = 0;
    unsigned int bgn = (qtype == XKRT_QUEUE_TYPE_ALL) ?                   0 : qtype;
    unsigned int end = (qtype == XKRT_QUEUE_TYPE_ALL) ? XKRT_QUEUE_TYPE_ALL : qtype + 1;
    for (unsigned int s = bgn ; s < end ; ++s)
    {
        for (int i = 0 ; i < device->count[s] ; ++i)
        {
            command_queue_t * queue = device->queues[tid][s][i];
            assert(queue);

            if (queue->pending.is_empty())
                continue ;

            xkrt_command_queue_list_counter_t n;
            do {
                if (blocking)
                {
                    this->device_command_queue_pending_wait(device, queue);
                    err = 0;
                }
                else
                    err = this->device_command_queue_pending_progress(device, queue);
                n = queue->pending.size();
                assert(n < queue->pending.capacity);
            } while (n > device->conf->offloader.queues[s].concurrency);
            assert(err == 0 || err == EINPROGRESS);
        }
    }
    return 0;
}

template int driver_t::device_offloader_progress<false>(device_t * device, int tid, const command_queue_type_t qtype);
template int driver_t::device_offloader_progress<true> (device_t * device, int tid, const command_queue_type_t qtype);

int
driver_t::device_offloader_wait_random_command(
    device_t * device,
    int tid
) {
    static thread_local unsigned int seed = 0x42;

    // randomly pick a type and a queue
    static_assert(XKRT_QUEUE_TYPE_ALL > 0);
    unsigned int rtype  = rand_r(&seed);
    unsigned int rqueue = rand_r(&seed);
    for (unsigned int ctype = 0 ; ctype < XKRT_QUEUE_TYPE_ALL ; ++ctype)
    {
        unsigned int s = (rtype + ctype) % XKRT_QUEUE_TYPE_ALL;

        command_queue_t * queue = NULL;
        for (int iqueue = 0 ; iqueue < device->count[s] ; ++iqueue)
        {
            unsigned int i = (rqueue + iqueue) % device->count[s];

            queue = device->queues[tid][s][i];
            assert(queue);

            // if the queue has pending commands
            if (!queue->pending.is_empty())
            {
                const xkrt_command_queue_list_counter_t i = queue->pending.pos.r;
                assert(i >= 0);
                assert(i < queue->pending.capacity);

                command_t * cmd = queue->pending.cmd + i;
                assert(cmd);

                assert(this->f_command_queue_wait);

                // waiting on the first event of the randomly elected queue
                int err = this->f_command_queue_wait(queue, cmd, i);

                // calling this to complete events and move queues pointers
                // but also detect out-of-order completions
                this->device_offloader_progress(device, tid);

                return err;
            }
        }
    }
    return 0;
}

//////////////////////
// QUEUE MANAGEMENT //
//////////////////////

int
driver_t::device_command_queue_pending_wait(
    device_t * device,
    command_queue_t * queue
) {
    assert(device->driver_type == this->type);
    if (!queue->pending.is_empty())
    {
        assert(this->f_command_queue_wait_all);
        this->f_command_queue_wait_all(queue);
        queue->complete_commands(queue->pending.pos.w);
    }
    return 0;
}

int
driver_t::device_command_queue_pending_progress(
    device_t * device,
    command_queue_t * queue
) {
    assert(device->driver_type == this->type);
    if (queue->pending.is_empty())
        return 0;

    LOGGER_DEBUG("Progressing pending commands of queue %p of type `%s` (%d pending) - ptr at r=%u, w=%u",
            queue, command_queue_type_to_str(queue->type), queue->pending.size(), queue->pending.pos.r, queue->pending.pos.w);
    // ask for progression of the given commands
    assert(this->f_command_queue_progress);
    const int r = this->f_command_queue_progress(queue);

    // move reading position to first uncompleted cmd
    const xkrt_command_queue_list_counter_t p = queue->pending.iterate([&] (xkrt_command_queue_list_counter_t p) {
        return queue->completed[p];
    });
    queue->pending.pos.r = p;

    LOGGER_DEBUG("Progressed pending commands of queue %p of type `%s` (%d pending)",
            queue, command_queue_type_to_str(queue->type), queue->pending.size());

    // return err code
    return r;
}


XKRT_NAMESPACE_END;
