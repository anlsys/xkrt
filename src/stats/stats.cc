/*
** Copyright 2024,2025 INRIA
**
** Contributors :
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
# if XKRT_SUPPORT_STATS

# include <string.h>

# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>
# include <xkrt/namespace.h>
# include <xkrt/runtime.h>
# include <xkrt/task/format.h>
# include <xkrt/task/task.hpp>

XKRT_NAMESPACE_USE;

typedef struct  device_stats_t
{
    struct {
        stats_int_t freed;
        struct {
            stats_int_t total;
            stats_int_t currently;
        } allocated;
        stats_int_t registered;
        stats_int_t unregistered;
        struct {
            struct {
                stats_int_t device;
                stats_int_t host;
            } advised;
            struct {
                stats_int_t device;
                stats_int_t host;
            } prefetched;
        } unified;
    } memory;

    struct {
        stats_int_t n;
        stats_int_t transfered;
    } queues[XKRT_QUEUE_TYPE_ALL];

    struct {
        stats_int_t commited;
        stats_int_t completed;
    } commands[ocg::COMMAND_TYPE_MAX];

}               device_stats_t;

static void
stats_device_agg_gather(runtime_t * runtime, device_stats_t * agg)
{
    agg->memory.registered   += runtime->stats.memory.registered;
    agg->memory.unregistered += runtime->stats.memory.unregistered;

    agg->memory.unified.advised.device += runtime->stats.memory.unified.advised.device;
    agg->memory.unified.advised.host   += runtime->stats.memory.unified.advised.host;

    agg->memory.unified.prefetched.device += runtime->stats.memory.unified.prefetched.device;
    agg->memory.unified.prefetched.host   += runtime->stats.memory.unified.prefetched.host;
}

static void
stats_device_agg(device_stats_t * src, device_stats_t * agg)
{
    agg->memory.freed += src->memory.freed;
    agg->memory.allocated.total += src->memory.allocated.total;
    agg->memory.allocated.currently += src->memory.allocated.currently;

    for (int stype = 0 ; stype < XKRT_QUEUE_TYPE_ALL ; ++stype)
    {
        agg->queues[stype].n += src->queues[stype].n;
        agg->queues[stype].transfered += src->queues[stype].transfered;
    }

    for (int cmd_type = 0 ; cmd_type < ocg::COMMAND_TYPE_MAX ; ++cmd_type)
    {
        agg->commands[cmd_type].commited += src->commands[cmd_type].commited;
        agg->commands[cmd_type].completed += src->commands[cmd_type].completed;
    }
}

static void
stats_device_report(device_stats_t * stats)
{
    char buffer[128];

    LOGGER_WARN("  Memory");

    # if 0
    metric_byte(buffer, sizeof(buffer), stats->memory.allocated.total.load());
    LOGGER_WARN("    Allocated (total): %s", buffer);

    metric_byte(buffer, sizeof(buffer), stats->memory.allocated.currently.load());
    LOGGER_WARN("    Allocated (currently): %s", buffer);

    metric_byte(buffer, sizeof(buffer), stats->memory.freed.load());
    LOGGER_WARN("    Freed: %s", buffer);
    # else
    LOGGER_WARN("    Allocated (total): %zuB", stats->memory.allocated.total.load());
    LOGGER_WARN("    Allocated (currently): %zuB", stats->memory.allocated.currently.load());
    LOGGER_WARN("    Freed: %zuB", stats->memory.freed.load());
    # endif

    if (stats->memory.registered.load() || stats->memory.unregistered.load())
    {
        metric_byte(buffer, sizeof(buffer), stats->memory.registered.load());
        LOGGER_WARN("    Registered: %s", buffer);

        metric_byte(buffer, sizeof(buffer), stats->memory.unregistered.load());
        LOGGER_WARN("    Unregistered: %s", buffer);
    }

    if (stats->memory.unified.advised.device.load() || stats->memory.unified.advised.host.load())
    {
        metric_byte(buffer, sizeof(buffer), stats->memory.unified.advised.device.load());
        LOGGER_WARN("    Unified Advised Device: %s", buffer);

        metric_byte(buffer, sizeof(buffer), stats->memory.unified.advised.host.load());
        LOGGER_WARN("    Unified Advised Host: %s", buffer);
    }

    if (stats->memory.unified.prefetched.device.load() || stats->memory.unified.prefetched.host.load())
    {
        metric_byte(buffer, sizeof(buffer), stats->memory.unified.prefetched.host.load());
        LOGGER_WARN("    Unified Prefetched Device: %s", buffer);

        metric_byte(buffer, sizeof(buffer), stats->memory.unified.prefetched.host.load());
        LOGGER_WARN("    Unified Prefetched Host: %s", buffer);
    }

    LOGGER_WARN("  Queues");
    for (int stype = 0 ; stype < XKRT_QUEUE_TYPE_ALL ; ++stype)
    {
        # if 0
        metric_byte(buffer, sizeof(buffer), stats->queues[stype].transfered.load());
        LOGGER_WARN("    `%4s` - with %2lu queues - transfered %s", ocg::command_type_to_str((queue_type_t) stype), stats->queues[stype].n.load(), buffer);
        # else
        LOGGER_WARN("    `%8s` - with %2lu queues - transfered %zuB", command_queue_type_to_str((command_queue_type_t) stype), stats->queues[stype].n.load(), stats->queues[stype].transfered.load());
        # endif
    }

    LOGGER_WARN("  Commands");
    for (int cmd_type = 0 ; cmd_type < ocg::COMMAND_TYPE_MAX ; ++cmd_type)
    {
        LOGGER_WARN(
            "    `%13s` - commited %6zu - completed %6zu",
            ocg::command_type_to_str((ocg::command_type_t) cmd_type),
            stats->commands[cmd_type].commited.load(),
            stats->commands[cmd_type].completed.load()
        );
    }
}

static void
stats_device_gather(
    device_t * device,
    device_stats_t * stats
) {
    memset(stats, 0, sizeof(device_stats_t));

    stats->memory.freed = device->stats.memory.freed.load();
    stats->memory.allocated.total = device->stats.memory.allocated.total.load();
    stats->memory.allocated.currently = device->stats.memory.allocated.currently.load();

    int nthreads = device->team->get_nthreads();
    for (int device_tid = 0 ; device_tid < nthreads ; ++device_tid)
    {
        for (int stype = 0 ; stype < XKRT_QUEUE_TYPE_ALL ; ++stype)
        {
            for (int queue_id = 0 ; queue_id < device->count[stype] ; ++queue_id)
            {
                command_queue_t * queue = device->queues[device_tid][stype][queue_id];
                for (int cmd_type = 0 ; cmd_type < ocg::COMMAND_TYPE_MAX ; ++cmd_type)
                {
                    stats->commands[cmd_type].commited += queue->stats.commands[cmd_type].commited.load();
                    stats->commands[cmd_type].completed += queue->stats.commands[cmd_type].completed.load();
                }
                stats->queues[stype].transfered += queue->stats.transfered.load();
            }
            stats->queues[stype].n += device->count[stype];
        }
    }
}

static void
stats_tasks_report(runtime_t * runtime)
{
    uint64_t total_edges_skipped = runtime->stats.edges.skipped.load();
    uint64_t total_edges_set     = runtime->stats.edges.set.load();
    uint64_t total_tasks         = 0;

    bool dumped = false;
    for (size_t i = 0 ; i < XKRT_TASK_FORMAT_MAX ; ++i)
    {
        task_format_t * format = runtime->task_format_get((task_format_id_t) i);
        if (format == NULL)
            break ;

        uint64_t c1 = runtime->stats.tasks[i].commited.load();
        uint64_t c2 = runtime->stats.tasks[i].submitted.load();
        uint64_t c3 = runtime->stats.tasks[i].completed.load();

        if (c1 || c2 || c3)
        {
            if (!dumped)
            {
                dumped = true;
                LOGGER_WARN("  Per format");
            }
            LOGGER_WARN("  `%16s` - %6zu commited - %6zu submitted - %6zu completed", format->label, c1, c2, c3);
        }

        total_tasks += c1;
    }

    # if XKRT_SUPPORT_DEBUG
    LOGGER_WARN("  On the main thread");
    thread_t * thread = thread_t::get_tls();
    int counter[TASK_STATE_MAX];
    memset(counter, 0, sizeof(counter));
    for (task_t * & task : thread->tasks)
    {
        assert(task->state.value >= TASK_STATE_ALLOCATED && task->state.value < TASK_STATE_MAX);
        ++counter[task->state.value];
    }

    for (int i = 0 ; i < TASK_STATE_MAX ; ++i)
        LOGGER_WARN("  `%8d` tasks in state `%12s`", counter[i], xkrt_task_state_to_str((task_state_t)i));

    # endif /* XKRT_SUPPORT_DEBUG */

    LOGGER_WARN("  Total: %zu tasks, %zu data-handle edges set, %zu data-handle edges skipped",
            total_tasks, total_edges_set, total_edges_skipped);
}

void
runtime_t::stats_report(void)
{
    LOGGER_WARN("----------------- STATS -----------------");
    device_stats_t agg;
    memset(&agg, 0, sizeof(agg));

    for (device_unique_id_t device_unique_id = 0 ; device_unique_id < this->drivers.devices.n ; ++device_unique_id)
    {
        device_t * device = this->drivers.devices.list[device_unique_id];

        driver_t * driver = this->driver_get(device->driver_type);
        LOGGER_WARN("Device %u", device->unique_id);

        char info[512];
        driver->f_device_info(device->driver_id, info, sizeof(info));
        LOGGER_WARN("  Info: %s", info);

        device_stats_t stats;
        stats_device_gather(device, &stats);
        stats_device_report(&stats);
        stats_device_agg(&stats, &agg);
    }
    stats_device_agg_gather(this, &agg);

    LOGGER_WARN("-----------------------------------------");
    LOGGER_WARN("All Devices");
    stats_device_report(&agg);
    LOGGER_WARN("-----------------------------------------");
    LOGGER_WARN("Tasks");
    stats_tasks_report(this);
    LOGGER_WARN("-----------------------------------------");
}

# endif /* XKRT_SUPPORT_STATS */
