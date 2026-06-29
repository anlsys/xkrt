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

# include <xkrt/runtime.h>
# include <xkrt/internals.h>
# include <xkrt/conf/conf.h>
# include <xkrt/logger/logger.h>
# include <xkrt/driver/driver.h>
# include <xkrt/memory/alignedas.h>
# include <xkrt/sync/spinlock.h>

# include <atomic>
# include <stdlib.h>
# include <string.h>
# include <signal.h>

# include <hwloc.h>

XKRT_NAMESPACE_BEGIN

//////////////////////////////
//  Runtime initialization  //
//////////////////////////////

static inline void
task_format_register(runtime_t * runtime)
{
    runtime->task_formats_init();
    task_host_capture_register_format(runtime);
    memory_copy_async_register_format(runtime);
    memory_register_async_register_format(runtime);
    file_async_register_format(runtime);
}

int
runtime_t::init(void)
{
    LOGGER_INFO("Initializing XKRT");

    // initialize TLS
    thread_t::get_tls();

    # if XKRT_SUPPORT_STATS
    memset(&this->stats, 0, sizeof(this->stats));
    new (&this->stats.tasks) memory_pool_t<task_stats_t>();
    # endif /* XKRT_SUPPORT_STATS */

    // set affinities to 0
    memset(&this->router.affinity, 0, sizeof(this->router.affinity));

    // create topology
    hwloc_topology_init(&this->topology);
    hwloc_topology_load(this->topology);

    // load
    this->conf.init();
    task_format_register(this);

    // the '+1' is to enforce the host device, always
    drivers_init(this);

    # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
    // initialize the registered memory map
    if (this->conf.protect_registered_memory_overflow)
    {
        new (&this->registered_memory) std::map<uintptr_t, size_t>();
    }
    else
    {
        LOGGER_WARN("Compiled with `MEMORY_REGISTER_OVERFLOW_PROTECTION` but `XKAAPI_MEMORY_REGISTER_PROTECT_OVERFLOW` environment variable is not set");
    }
    # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

    // mark runtime as initialized
    this->state = runtime_t::state_t::INITIALIZED;

    return 0;
}

int
runtime_t::deinit(void)
{
    LOGGER_INFO("Deinitializing XKRT");
    assert(this->state == runtime_t::state_t::INITIALIZED);

    # if XKRT_SUPPORT_STATS
    if (this->conf.report_stats_on_deinit)
        this->stats_report();
    # endif /* XKRT_SUPPORT_STATS */

    this->state = runtime_t::state_t::DEINITIALIZED;
    drivers_deinit(this);
    hwloc_topology_destroy(this->topology);
    this->formats.all.list.release();
    # if XKRT_SUPPORT_STATS
    this->stats.tasks.release();
    # endif /* XKRT_SUPPORT_STATS */

    return 0;
}

///////////////
// UTILITIES //
///////////////

unsigned int
runtime_t::get_ndevices_max(void)
{
    unsigned int count = 1;
    for (int i = 0 ; i < XKRT_DRIVER_TYPE_MAX ; ++i)
    {
        if (i != XKRT_DRIVER_TYPE_HOST)
        {
            driver_t * driver = this->driver_get((driver_type_t) i);
            if (driver && driver->f_get_ndevices_max)
                count += driver->f_get_ndevices_max();
        }
    }

    return count;
}

unsigned int
runtime_t::get_ndevices(void)
{
    return this->drivers.devices.n;
}

XKRT_NAMESPACE_END
