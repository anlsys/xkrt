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
# include <xkrt/memory/access/blas/memory-tree.hpp>

# if XKRT_MEMORY_REGISTER_PAGE
#  include <xkrt/memory/pageas.h>
# endif /* XKRT_MEMORY_REGISTER_PAGE */

XKRT_NAMESPACE_BEGIN

///////////////////////////////////
//  ORIGINAL KAAPI 1.0 INTERFACE //
///////////////////////////////////

//  General idea of these interfaces
//      - Nvidia GPUs serializes memory pinning anyway
//      - We have a single thread dedicated to pinning memory to any device
//      - it schedule 'pinning tasks' that are tasks with read/write access on the memory


//  Some ideas:
//      - can we lock pages in memory independently from CUDA / HIP / ... via
//      'mlock' and notify the driver that the memory is locked ? Would be
//      better in case of server with different GPU vendors...


# pragma message(TODO "The current implementation spawn independent tasks. Maybe make it dependent to a specific type of access")

# if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
static inline task_t *
__memory_register_get_memory_controller_task(runtime_t * runtime)
{
    thread_t * tls = thread_t::get_tls();
    assert(tls);

    task_t * task = tls->current_task;
    assert(task);

    /* if registering within a registration task, then retrieve parent memory controller */
    if (task->fmtid == runtime->formats.memory_register_async ||
            task->fmtid == runtime->formats.host_capture)
    {
        assert(task->parent);
        assert(task->parent->flags & TASK_FLAG_DOMAIN);
        task = task->parent;
    }
    assert(task && (task->flags & TASK_FLAG_DOMAIN));
    return task;
}
# endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

int
runtime_t::memory_register(
    void * ptr,
    size_t size
) {
    # if XKRT_MEMORY_REGISTER_PAGE
    if (this->conf.protect_registered_memory_overflow)
        pageas(ptr, size, (uintptr_t *) &ptr, &size);
    # endif /* XKRT_MEMORY_REGISTER_PAGE */

    # if XKRT_MEMORY_REGISTER_ASSISTED
    const uintptr_t a = (const uintptr_t) ptr;
    const uintptr_t b = a + size;
    # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

    # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
    /* notify the current memory coherency controllers that the memory got registered */
    task_t * task = __memory_register_get_memory_controller_task(this);
    task_dom_info_t * dom = TASK_DOM_INFO(task);
    assert(dom);
    # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

    /* register the memory segment in each driver */
    for (uint8_t driver_id = 0 ; driver_id < XKRT_DRIVER_TYPE_MAX; ++driver_id)
    {
        driver_t * driver = this->driver_get((driver_type_t) driver_id);
        if (!driver)
            continue ;
        if (!driver->f_memory_host_register)
            LOGGER_DEBUG("Driver `%u` does not implement memory register", driver_id);
        else
        {
            # if XKRT_MEMORY_REGISTER_ASSISTED
            std::vector<Interval> intervals;
            this->registered_pages.find(a, b, intervals);
            assert(std::is_sorted(intervals.begin(), intervals.end()));

            unsigned int i = 0;
            uintptr_t p1 = a;

            while (1)
            {
                uintptr_t p2 = (i == intervals.size()) ? b : intervals[i].a;
                if (p1 < p2)
                {
                    ptr  = (void *) p1;
                    size = p2 - p1;
                    LOGGER_DEBUG("Registering [%lu..%lu] to driver %u", (uintptr_t) ptr, ((uintptr_t) ptr) + size, driver_id);
            # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

                    if (driver->f_memory_host_register(ptr, size))
                        LOGGER_ERROR("Could not register memory for driver `%s`", driver->f_get_name());

                    # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
                    if (this->conf.protect_registered_memory_overflow)
                    {
                        /* save in the registered map, for later accesses, that may use
                         * different memory controllers */
                        this->registered_memory[(uintptr_t)ptr] = size;

                        if (dom->mccs.interval)
                            ((BLASMemoryTree *) dom->mccs.interval)->registered((uintptr_t)ptr, size);

                        for (MemoryCoherencyController * mcc : dom->mccs.blas)
                            ((BLASMemoryTree *)mcc)->registered((uintptr_t)ptr, size);
                    }
                    # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

                    # if XKRT_SUPPORT_STATS
                    this->stats.memory.registered += size;
                    # endif /* XKRT_SUPPORT_STATS */

            # if XKRT_MEMORY_REGISTER_ASSISTED
                }
                if (i == intervals.size())
                    break ;
                p1 = intervals[i].b;
                ++i;
            }
            # endif /* XKRT_MEMORY_REGISTER_ASSISTED */
        }
    }

    # if XKRT_MEMORY_REGISTER_ASSISTED
    this->registered_pages.fill(a, b);
    # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

    return 0;
}

int
runtime_t::memory_unregister(
    void * ptr,
    size_t size
) {
    # if XKRT_MEMORY_REGISTER_PAGE
    if (this->conf.protect_registered_memory_overflow)
        pageas(ptr, size, (uintptr_t *) &ptr, &size);
    # endif /* XKRT_MEMORY_REGISTER_PAGE */

    # if XKRT_MEMORY_REGISTER_ASSISTED
    const uintptr_t a = (const uintptr_t) ptr;
    const uintptr_t b = a + size;
    # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

    # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
    /* notify the current memory coherency controllers that the memory got unregistered */
    task_t * task = __memory_register_get_memory_controller_task(this);
    task_dom_info_t * dom = TASK_DOM_INFO(task);
    assert(dom);
    # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

    /* register the memory segment in each driver */
    for (uint8_t driver_id = 0 ; driver_id < XKRT_DRIVER_TYPE_MAX; ++driver_id)
    {
        driver_t * driver = this->driver_get((driver_type_t) driver_id);
        if (!driver)
            continue ;
        if (!driver->f_memory_host_unregister)
            LOGGER_DEBUG("Driver `%u` does not implement memory unregister", driver_id);
        else
        {
            # if XKRT_MEMORY_REGISTER_ASSISTED
            std::vector<Interval> intervals;
            this->registered_pages.find(a, b, intervals);
            //assert(std::is_sorted(intervals.begin(), intervals.end()));
            std::sort(intervals.begin(), intervals.end());

            for (Interval & interval : intervals)
            {
                ptr  = (void *) interval.a;
                size = interval.length();
                LOGGER_DEBUG("Unregistering [%lu..%lu] to driver %u", (uintptr_t) ptr, ((uintptr_t) ptr) + size, driver_id);
            # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

                if (driver->f_memory_host_unregister(ptr, size))
                    LOGGER_ERROR("Could not unregister memory for driver `%s`", driver->f_get_name());

                # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
                if (this->conf.protect_registered_memory_overflow)
                {
                    /* save in the registered map, for later accesses, that may use
                     * different memory controllers */
                    this->registered_memory.erase((uintptr_t)ptr);

                    if (dom->mccs.interval)
                        ((BLASMemoryTree *) dom->mccs.interval)->unregistered((uintptr_t)ptr, size);

                    for (MemoryCoherencyController * mcc : dom->mccs.blas)
                        ((BLASMemoryTree *)mcc)->unregistered((uintptr_t)ptr, size);
                }
                # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

                # if XKRT_SUPPORT_STATS
                this->stats.memory.unregistered += size;
                # endif /* XKRT_SUPPORT_STATS */

            # if XKRT_MEMORY_REGISTER_ASSISTED
            }
            # endif /* XKRT_MEMORY_REGISTER_ASSISTED */
        }
    }

    # if XKRT_MEMORY_REGISTER_ASSISTED
    this->registered_pages.unfill(a, b);
    # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

    return 0;
}

int
runtime_t::memory_register_async(void * ptr, size_t size)
{
    // TODO : could be optimized using a custom format for register tasks
    this->task_spawn(
        [=] (runtime_t * runtime, device_t * device, task_t * task) {
            (void) device; (void) task;
            runtime->memory_register(ptr, size);
        }
    );
    return 0;
}

int
runtime_t::memory_unregister_async(void * ptr, size_t size)
{
    // TODO : could be optimized using a custom format for unregister tasks
    this->task_spawn(
        [=] (runtime_t * runtime, device_t * device, task_t * task) {
            (void) device; (void) task;
            runtime->memory_unregister(ptr, size);
        }
    );
    return 0;
}

///////////////////////////////
//  NEW XKAAPI 2.0 INTERFACE //
///////////////////////////////

typedef struct  memory_op_async_args_t
{
    uintptr_t start;
    uintptr_t end;

}               memory_op_async_args_t;

typedef enum    memory_op_type_t
{
    REGISTER,
    UNREGISTER,
    TOUCH
}               memory_op_type_t;

constexpr size_t args_size = sizeof(memory_op_async_args_t);
constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT;

template<memory_op_type_t T>
static void
body_memory_async(runtime_t * runtime, device_t * device, task_t * task)
{
    (void) device;
    assert(runtime);
    assert(task);

    constexpr task_access_counter_t AC = (T == TOUCH) ? 1 : 2;
    constexpr size_t task_size = task_compute_size(flags, AC);

    memory_op_async_args_t * args = (memory_op_async_args_t *) TASK_ARGS(task, task_size);
    assert(args->start < args->end);

    if constexpr (T == REGISTER)
        runtime->memory_register((void *) args->start, (size_t) (args->end - args->start));
    else if constexpr (T == UNREGISTER)
        runtime->memory_unregister((void *) args->start, (size_t) (args->end - args->start));
    else if constexpr (T == TOUCH)
    {
        // volatile to trick the compiler and avoid optimization of *p = *p
        volatile unsigned char * a = (volatile unsigned char *) args->start;
           const unsigned char * b = (   const unsigned char *) args->end;
             const size_t pagesize = (size_t) getpagesize();

        for ( ; a < b ; a += pagesize)
            *a = 0;
    }
}

template<memory_op_type_t T>
static int
memory_op_async(
    runtime_t * runtime,
    team_t * team,
    void * ptr,
    size_t size,
    int n
) {
    assert(n > 0);

    constexpr task_access_counter_t AC = (T == TOUCH) ? 1 : 2;
    constexpr size_t task_size = task_compute_size(flags, AC);

    thread_t * tls = thread_t::get_tls();
    assert(tls);

    const task_format_id_t fmtid = (T == REGISTER)   ? runtime->formats.memory_register_async   :
                                   (T == UNREGISTER) ? runtime->formats.memory_unregister_async :
                                   (T == TOUCH)      ? runtime->formats.memory_touch_async      :
                                   0;
    assert(fmtid);

    const size_t pagesize = (size_t) getpagesize();

    // compute number of pages to register
    const size_t npages = (size < pagesize) ? 1 : size / pagesize;

    // compute number of tasks
    const size_t ntasks = (npages < (size_t) n) ? npages : (size_t) n;

    // compute number of pages to register per task
    const size_t pages_per_task = (npages < ntasks) ? 1 : npages / ntasks;

    // round to closest pages
    const uintptr_t  p = (const uintptr_t) ptr;
    const uintptr_t pp = p + size;
    const uintptr_t  a = p - (p % pagesize);
    const uintptr_t  b = pp + (pagesize - (pp % pagesize)) % pagesize;
    assert(a < b);
    assert(a % pagesize == 0);
    assert(b % pagesize == 0);

    // spawn tasks
    for (size_t i = 0 ; i < ntasks ; ++i)
    {
        // create a task that will register/pin/unpin the memory
        task_t * task = runtime->task_new(fmtid, flags, task_size + args_size);

        task_dep_info_t * dep = TASK_DEP_INFO(task);
        new (dep) task_dep_info_t(AC);

        // setup register args
        memory_op_async_args_t * args = (memory_op_async_args_t *) TASK_ARGS(task);

        // ensure the same page is not registered twice by consecutive tasks
        args->start = a + (i+0) * pagesize * pages_per_task;
        args->end   = a + (i+1) * pagesize * pages_per_task;

        // clamp upper
        if (args->end > p + size)
            args->end = p + size;

        assert(a <= args->start);
        assert(     args->start < b);
        assert(a <= args->end);
        assert(     args->end <= b);
        assert(args->start < args->end);

        // virtual write onto the memory segment
        access_t * accesses = TASK_ACCESSES(task, flags);
        new (accesses + 0) access_t(task, args->start, args->end, ACCESS_MODE_VW);

        // if register/unregister, create a virtual write on NULL, to
        // serialize, and avoid blocking thread in cuda driver
        if constexpr(T == REGISTER || T == UNREGISTER)
            new (accesses + 1) access_t(task, (const void *) NULL, ACCESS_MODE_VW, ACCESS_CONCURRENCY_COMMUTATIVE);

        # if XKRT_SUPPORT_DEBUG
        snprintf(task->label, sizeof(task->label),
                T == REGISTER   ? "register"   :
                T == UNREGISTER ? "unregister" :
                T == TOUCH      ? "touch"      :
                "(null)");
        # endif /* XKRT_SUPPORT_DEBUG */

        // resolve dependencies
        tls->resolve(accesses, AC);

        // commit task
        tls->commit(task, runtime_t::task_team_enqueue, runtime, team);
    }

    return 0;
}

int
runtime_t::memory_unregister_async(
    team_t * team,
    void * ptr,
    const size_t size,
    int n
) {
    return memory_op_async<UNREGISTER>(this, team, ptr, size, n);
}

int
runtime_t::memory_register_async(
    team_t * team,
    void * ptr,
    const size_t size,
    int n
) {
    return memory_op_async<REGISTER>(this, team, ptr, size, n);
}

int
runtime_t::memory_touch_async(
    team_t * team,
    void * ptr,
    const size_t size,
    int n
) {
    return memory_op_async<TOUCH>(this, team, ptr, size, n);
}

void
memory_register_async_register_format(runtime_t * runtime)
{
    {
        task_format_t format;
        memset(format.f, 0, sizeof(format.f));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_memory_async<REGISTER>;
        snprintf(format.label, sizeof(format.label), "memory_register_async");
        runtime->formats.memory_register_async = runtime->task_format_create(&format);
    }

    {
        task_format_t format;
        memset(format.f, 0, sizeof(format.f));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_memory_async<UNREGISTER>;
        snprintf(format.label, sizeof(format.label), "memory_unregister_async");
        runtime->formats.memory_unregister_async = runtime->task_format_create(&format);
    }

    {
        task_format_t format;
        memset(format.f, 0, sizeof(format.f));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) body_memory_async<TOUCH>;
        snprintf(format.label, sizeof(format.label), "memory_touch_async");
        runtime->formats.memory_touch_async = runtime->task_format_create(&format);
    }
}

XKRT_NAMESPACE_END
