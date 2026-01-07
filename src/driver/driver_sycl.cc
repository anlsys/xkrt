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

# define XKRT_DRIVER_ENTRYPOINT(N) XKRT_DRIVER_TYPE_SYCL_ ## N

# include <xkrt/runtime.h>
# include <xkrt/support.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/driver-sycl.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/logger-hwloc.h>
# include <xkrt/logger/metric.h>
# include <xkrt/sync/bits.h>
# include <xkrt/sync/mutex.h>

# include <hwloc.h>
// # include <hwloc/cuda.h>
# include <hwloc/glibc-sched.h>

//  this flag skips sycl/ur and jumps straight to ZE on interfaces that have
//  poor implementations
# if XKRT_SUPPORT_ZE
#  define BYPASS_SYCL 0
#  if BYPASS_SYCL
#   include <ze_api.h>
#   include <xkrt/logger/logger-ze.h>
#  endif /* BYPASS_SYCL */
# else
#  define BYPASS_SYCL 0
# endif /* XKRT_SUPPORT_ZE */

# include <cassert>
# include <cstdio>
# include <cstdint>
# include <cerrno>

# include <optional>

XKRT_NAMESPACE_BEGIN

/* number of used device for this run */
static std::optional<device_sycl_t> DEVICES[XKRT_DEVICES_MAX];

/* maximum number of devices installed */
static int ndevices_max;

static inline device_t *
device_get(device_driver_id_t device_driver_id)
{
    return (device_t *) (DEVICES + device_driver_id);
}

static inline device_sycl_t *
device_sycl_get(device_driver_id_t device_driver_id)
{
    return (device_sycl_t *) device_get(device_driver_id);
}

static unsigned int
XKRT_DRIVER_ENTRYPOINT(get_ndevices_max)(void)
{
    return ndevices_max;
}

static int
XKRT_DRIVER_ENTRYPOINT(init)(
    unsigned int ndevices,
    bool use_p2p
) {
    assert(ndevices);
    # pragma message(TODO "Support sycl drvier 'use_p2p'")
    (void) use_p2p;

    try
    {
        auto platforms = sycl::platform::get_platforms();
        int i = 0;
        for (const auto & platform : platforms)
        {
            auto devices = platform.get_devices();
            ndevices_max += devices.size();
            if (i < ndevices)
            {
                for (const auto & dev : devices)
                {
                    if (dev.is_gpu())
                    {
                        device_sycl_t * device = device_sycl_get(i);
                        device->sycl.platform = platform;
                        device->sycl.device = dev;
                        if (++i == ndevices)
                            break ;
                    }
                }
            }
        }
    }
    catch (const sycl::exception &e)
    {
        std::cerr << "SYCL exception caught: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

static void
XKRT_DRIVER_ENTRYPOINT(finalize)(void)
{
}

static const char *
XKRT_DRIVER_ENTRYPOINT(get_name)(void)
{
    return "SYCL";
}

static int
XKRT_DRIVER_ENTRYPOINT(device_cpuset)(
    hwloc_topology_t topology,
    cpu_set_t * schedset,
    device_driver_id_t device_driver_id
) {
    device_sycl_t * device = device_sycl_get(device_driver_id);
    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();

    LOGGER_IMPL("Couldnt get cpuset of a SYCL device. Device threads will bind to any CPUs");

    # if 0
    hwloc_bitmap_fill(cpuset);
    // TODO : maybe check back-end and use the associated hwloc call - or find
    // a portable way to do that in SYCL
    CPU_ZERO(schedset);
    HWLOC_SAFE_CALL(hwloc_cpuset_to_glibc_sched_affinity(
                topology, cpuset, schedset, sizeof(cpu_set_t)));
    hwloc_bitmap_free(cpuset);
    # else
    CPU_ZERO(schedset);
    for (int i = device_driver_id * 4 ; i < (device_driver_id + 1) * 4 ; ++i)
        CPU_SET(i, schedset);
    # endif


    return 0;
}

static device_t *
XKRT_DRIVER_ENTRYPOINT(device_create)(
    driver_t * driver,
    device_driver_id_t device_driver_id
) {
    device_sycl_t * device = device_sycl_get(device_driver_id);
    assert(device->inherited.state == XKRT_DEVICE_STATE_DEALLOCATED);
    new (&device->sycl.alloc_queue) sycl::queue(device->sycl.device);
    return (device_t *) device;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_init)(device_driver_id_t device_driver_id)
{
    device_sycl_t * device = device_sycl_get(device_driver_id);
    assert(device->inherited.state == XKRT_DEVICE_STATE_CREATE);
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_device_allocate)(
    device_driver_id_t device_driver_id,
    const size_t size,
    int area_idx
) {
    assert(area_idx == 0);
    device_sycl_t * device = device_sycl_get(device_driver_id);
    void * device_ptr = sycl::malloc_device(size, device->sycl.alloc_queue);
    return device_ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_deallocate)(
    device_driver_id_t device_driver_id,
    void * ptr,
    const size_t size,
    int area_idx
) {
    (void) size;
    assert(area_idx == 0);
    device_sycl_t * device = device_sycl_get(device_driver_id);
    sycl::free(ptr, device->sycl.alloc_queue);
}

# if 0

static void *
XKRT_DRIVER_ENTRYPOINT(memory_unified_allocate)(
    device_driver_id_t device_driver_id,
    const size_t size
) {
    (void) device_driver_id;
    (void) size;
    LOGGER_FATAL("Not implemented");
    return NULL;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_unified_deallocate)(
    device_driver_id_t device_driver_id,
    void * ptr,
    const size_t size
) {
    (void) device_driver_id;
    (void) ptr;
    (void) size;
    LOGGER_FATAL("Not implemented");
}

# endif

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_info)(
    device_driver_id_t device_driver_id,
    device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX],
    int * nmemories
) {
    device_sycl_t * device = device_sycl_get(device_driver_id);
    info[0].capacity = device->sycl.device.get_info<sycl::info::device::global_mem_size>();
    info[0].used     = (size_t) -1;
    strncpy(info[0].name, "(null)", sizeof(info[0].name));
    *nmemories = 1;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_destroy)(device_driver_id_t device_driver_id)
{
    device_sycl_t * device = device_sycl_get(device_driver_id);
    (void) device;
    return 0;
}

/* Called for each device of the driver once they all have been initialized */
static int
XKRT_DRIVER_ENTRYPOINT(device_commit)(
    device_driver_id_t device_driver_id,
    device_global_id_bitfield_t * affinity
) {
    assert(affinity);

    device_sycl_t * device = device_sycl_get(device_driver_id);
    device_global_id_t device_global_id = device->inherited.global_id;

    LOGGER_IMPL("Set actual affinity, hardcoded for now");
    int rank = 0;
    affinity[rank++] = (1 << device_global_id);
    # if 1 // Aurora specific - set high affinity between the two stacks for the gpu
    device_global_id_t stack_id = (device_global_id % 2 == 0) ? (device_global_id + 1) : (device_global_id - 1);
    affinity[rank++] =  (1 << stack_id);
    # endif
    affinity[rank++] = ~affinity[0];

    return 0;
}

# if 0
static int
XKRT_DRIVER_ENTRYPOINT(memory_host_register)(
    void * ptr,
    uint64_t size
) {
    SYCL_SAFE_CALL(cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE));
    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_unregister)(
    void * ptr,
    uint64_t size
) {
    (void) size;
    SYCL_SAFE_CALL(cuMemHostUnregister(ptr));
    return 0;
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_host_allocate)(
    device_driver_id_t device_driver_id,
    uint64_t size
) {
    (void) device_driver_id;
    void * ptr;
    cu_set_context(device_driver_id);
    SYCL_SAFE_CALL(cuMemHostAlloc(&ptr, size, CU_MEMHOSTREGISTER_PORTABLE));
    // SYCL_SAFE_CALL(cuHostAlloc(&ptr, size, cuHostRegisterPortable | cuHostAllocWriteCombined));
    return ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_host_deallocate)(
    device_driver_id_t device_driver_id,
    void * mem,
    uint64_t size
) {
    (void) device_driver_id;
    (void) size;
    SYCL_SAFE_CALL(cuMemFreeHost(mem));
}

# endif

static int
XKRT_DRIVER_ENTRYPOINT(queue_suggest)(
    device_driver_id_t device_driver_id,
    queue_type_t qtype
) {
    (void) device_driver_id;

    switch (qtype)
    {
        case (XKRT_QUEUE_TYPE_KERN):
            return 8;
        default:
            return 4;
    }
}

static int
XKRT_DRIVER_ENTRYPOINT(queue_command_launch)(
    queue_t * iqueue,
    command_t * cmd,
    queue_command_list_counter_t idx
) {
    queue_sycl_t * queue = (queue_sycl_t *) iqueue;
    assert(queue);

    sycl::queue & q = queue->sycl.queue;
    sycl::event * e = queue->sycl.events.buffer + idx;

    switch (cmd->type)
    {
        case (COMMAND_TYPE_COPY_H2D_1D):
        case (COMMAND_TYPE_COPY_D2H_1D):
        case (COMMAND_TYPE_COPY_D2D_1D):
        {
            void * src = (void *) cmd->copy_1D.src_device_addr;
            void * dst = (void *) cmd->copy_1D.dst_device_addr;
            const size_t count  = cmd->copy_1D.size;
            assert(count > 0);

            sycl::event evt = q.memcpy(dst, src, count);
            new (e) sycl::event(evt);

            return EINPROGRESS;
        }

        case (COMMAND_TYPE_COPY_H2D_2D):
        case (COMMAND_TYPE_COPY_D2H_2D):
        case (COMMAND_TYPE_COPY_D2D_2D):
        {
                  void * dst    = (      void *) cmd->copy_2D.dst_device_view.addr;
            const void * src    = (const void *) cmd->copy_2D.src_device_view.addr;

            const size_t dst_pitch = cmd->copy_2D.dst_device_view.ld * cmd->copy_2D.sizeof_type;
            const size_t src_pitch = cmd->copy_2D.src_device_view.ld * cmd->copy_2D.sizeof_type;

            const size_t width  = cmd->copy_2D.m * cmd->copy_2D.sizeof_type;
            const size_t height = cmd->copy_2D.n;

            # if BYPASS_SYCL

            // get ze list
            ze_command_list_handle_t list;
            auto command_type = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q);
            if (auto * ptr_queue_handle = std::get_if<ze_command_list_handle_t>(&command_type))
            {
                list = *ptr_queue_handle;
            }
            else if (auto * ptr_queue_handle = std::get_if<ze_command_queue_handle_t>(&command_type))
            {
                LOGGER_FATAL("QUEUE IS NON-IMMEDIATE, fuck that");
            }

            const uint32_t dst_slice_pitch = 0;
            const ze_copy_region_t dst_region = {
                .originX = 0,
                .originY = 0,
                .originZ = 0,
                .width   = (uint32_t) width,
                .height  = (uint32_t) height,
                .depth   = 1
            };

            const uint32_t src_slice_pitch = 0;
            const ze_copy_region_t src_region = {
                .originX = 0,
                .originY = 0,
                .originZ = 0,
                .width   = (uint32_t) width,
                .height  = (uint32_t) height,
                .depth   = 1
            };

            const uint32_t num_wait_events = 0;
            ze_event_handle_t * wait_events = NULL;

            // create a ze event
            new (e) sycl::event();
            ze_event_handle_t event = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(*e);
            ZE_SAFE_CALL(zeEventHostReset(event));

            ZE_SAFE_CALL(
                zeCommandListAppendMemoryCopyRegion(
                    list,
                    dst,
                   &dst_region,
                    dst_pitch,
                    dst_slice_pitch,
                    src,
                   &src_region,
                    src_pitch,
                    src_slice_pitch,
                    event,
                    num_wait_events,
                    wait_events
                )
            );

            # elif SYCL_EXT_ONEAPI_MEMCPY2D == 1

            const std::vector<sycl::event> dependencies = {};
            sycl::event evt = q.ext_oneapi_memcpy2d(dst, dst_pitch, src, src_pitch, width, height, dependencies);
            new (e) sycl::event(evt);

            # else

            LOGGER_FATAL("SYCL does not support memcpy2D");

            # endif

            return EINPROGRESS;
        }

        default:
            return EINVAL;
    }

    /* unreachable code */
    LOGGER_FATAL("Unreachable code");
}

static inline int
XKRT_DRIVER_ENTRYPOINT(queue_commands_wait)(
    queue_t * iqueue
) {
    queue_sycl_t * queue = (queue_sycl_t *) iqueue;
    assert(queue);

    # if 0
    SYCL_SAFE_CALL(cuStreamSynchronize(queue->cu.handle.high));
    SYCL_SAFE_CALL(cuStreamSynchronize(queue->cu.handle.low));
    # endif

    LOGGER_FATAL("Not implemented");

    return 0;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(queue_command_wait)(
    queue_t * iqueue,
    command_t * cmd,
    queue_command_list_counter_t idx
) {
    queue_sycl_t * queue = (queue_sycl_t *) iqueue;
    queue->sycl.queue.wait();
    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(queue_commands_progress)(
    queue_t * iqueue
) {
    assert(iqueue);

    queue_sycl_t * queue = (queue_sycl_t *) iqueue;
    int r = 0;

    iqueue->pending.progress([&] (command_t * cmd, queue_command_list_counter_t p) {

        switch (cmd->type)
        {
            case (COMMAND_TYPE_KERN):
            case (COMMAND_TYPE_COPY_H2H_1D):
            case (COMMAND_TYPE_COPY_H2D_1D):
            case (COMMAND_TYPE_COPY_D2H_1D):
            case (COMMAND_TYPE_COPY_D2D_1D):
            case (COMMAND_TYPE_COPY_H2H_2D):
            case (COMMAND_TYPE_COPY_H2D_2D):
            case (COMMAND_TYPE_COPY_D2H_2D):
            case (COMMAND_TYPE_COPY_D2D_2D):
            {
                sycl::event * e = queue->sycl.events.buffer + p;
                auto status = e->get_info<sycl::info::event::command_execution_status>();
                if (status == sycl::info::event_command_status::complete)
                {
                    // explicitly call event destructor
                    e->~event();
                    iqueue->complete_command(p);
                }
                else
                    r = EINPROGRESS;

                break ;
            }

            default:
                LOGGER_FATAL("Wrong command");
        }

        return true;
    });

    return r;
}

static queue_t *
XKRT_DRIVER_ENTRYPOINT(queue_create)(
    device_t * idevice,
    queue_type_t type,
    queue_command_list_counter_t capacity
) {
    assert(idevice);

    uint8_t * mem = (uint8_t *) malloc(sizeof(queue_sycl_t) + capacity * sizeof(sycl::event));
    assert(mem);

    device_sycl_t * device = (device_sycl_t *) idevice;
    queue_sycl_t * queue = (queue_sycl_t *) mem;

    /*************************/
    /* init xkrt queue */
    /*************************/
    queue_init(
        (queue_t *) queue,
        type,
        capacity,
        XKRT_DRIVER_ENTRYPOINT(queue_command_launch),
        XKRT_DRIVER_ENTRYPOINT(queue_commands_progress),
        XKRT_DRIVER_ENTRYPOINT(queue_commands_wait),
        XKRT_DRIVER_ENTRYPOINT(queue_command_wait)
    );

    /*************************/
    /* do cu specific init */
    /*************************/

    /* events */
    queue->sycl.events.buffer   = (sycl::event *) (queue + 1);
    queue->sycl.events.capacity = capacity;

    // TODO : how to initialize queues depending on `type` ?
    new (&queue->sycl.queue) sycl::queue(device->sycl.device);

    return (queue_t *) queue;
}

static void
XKRT_DRIVER_ENTRYPOINT(queue_delete)(
    queue_t * iqueue
) {
    queue_sycl_t * queue = (queue_sycl_t *) iqueue;
    queue->sycl.queue.~queue();
    free(queue);
}

static inline void
_print_mask(char * buffer, ssize_t size, uint64_t v)
{
    for (int i = 0; i < size; ++i)
        buffer[size-1-i] = (v & (1ULL<<i)) ? '1' : '0';
}

void
XKRT_DRIVER_ENTRYPOINT(device_info)(
    device_driver_id_t device_driver_id,
    char * buffer,
    size_t size
) {
    device_sycl_t * device = device_sycl_get(device_driver_id);
    sycl::device & dev = device->sycl.device;
    auto device_type = dev.get_info<sycl::info::device::device_type>();

    std::string name = dev.get_info<sycl::info::device::name>();
    std::string vendor = dev.get_info<sycl::info::device::vendor>();
    std::string driver = dev.get_info<sycl::info::device::driver_version>();
    const char * const type = (device_type == sycl::info::device_type::cpu) ? "CPU" : (device_type == sycl::info::device_type::gpu) ? "GPU" :  (device_type == sycl::info::device_type::accelerator) ? "Accelerator" : "unkn";
    const size_t memsize = dev.get_info<sycl::info::device::global_mem_size>() / (1024 * 1024 * 1024);
    const unsigned int maxcomputeunits = dev.get_info<sycl::info::device::max_compute_units>();
    const unsigned int numcomputeunits = dev.get_info<sycl::info::device::max_compute_units>();

    snprintf(buffer, size, "%s (%s, %s, %s), sycl device: %i, %zuGB, %u/%u compute units, ",
        name.c_str(), vendor.c_str(), driver.c_str(), type,
        device->inherited.global_id,
        memsize,
        numcomputeunits,
        maxcomputeunits
    );
}

#if 0
driver_module_t
XKRT_DRIVER_ENTRYPOINT(module_load)(
    device_driver_id_t device_driver_id,
    uint8_t * bin,
    size_t binsize,
    driver_module_format_t format
) {
    (void) binsize;
    assert(format == XKRT_DRIVER_MODULE_FORMAT_NATIVE);
    cu_set_context(device_driver_id);
    driver_module_t module = NULL;
    SYCL_SAFE_CALL(cuModuleLoadData((CUmodule *) &module, bin));
    assert(module);
    return module;
}

void
XKRT_DRIVER_ENTRYPOINT(module_unload)(
    driver_module_t module
) {
    SYCL_SAFE_CALL(cuModuleUnload((CUmodule) module));
}

driver_module_fn_t
XKRT_DRIVER_ENTRYPOINT(module_get_fn)(
    driver_module_t module,
    const char * name
) {
    driver_module_fn_t fn = NULL;
    SYCL_SAFE_CALL(cuModuleGetFunction((CUfunction *) &fn, (CUmodule) module, name));
    assert(fn);
    return fn;
}
# endif

# if 0
void
XKRT_DRIVER_ENTRYPOINT(power_start)(device_driver_id_t device_driver_id, power_t * pwr)
{
    (void) device_driver_id;
    (void) pwr;
    LOGGER_FATAL("impl me");
}

void
XKRT_DRIVER_ENTRYPOINT(power_stop)(device_driver_id_t device_driver_id, power_t * pwr)
{
    (void) device_driver_id;
    (void) pwr;
    LOGGER_FATAL("impl me");
}

# endif

driver_t *
XKRT_DRIVER_ENTRYPOINT(create_driver)(void)
{
    driver_sycl_t * driver = (driver_sycl_t *) calloc(1, sizeof(driver_sycl_t));
    assert(driver);

    # define REGISTER(func) driver->super.f_##func = XKRT_DRIVER_ENTRYPOINT(func)

    REGISTER(init);
    REGISTER(finalize);

    REGISTER(get_name);
    REGISTER(get_ndevices_max);

    REGISTER(device_create);
    REGISTER(device_init);
    REGISTER(device_commit);
    REGISTER(device_destroy);

    REGISTER(device_info);

    REGISTER(memory_device_info);
    REGISTER(memory_device_allocate);
    REGISTER(memory_device_deallocate);
    // REGISTER(memory_host_allocate);
    // REGISTER(memory_host_deallocate);
    // REGISTER(memory_host_register);
    // REGISTER(memory_host_unregister);
    // REGISTER(memory_unified_allocate);
    // REGISTER(memory_unified_deallocate);

    REGISTER(device_cpuset);

    REGISTER(queue_suggest);
    REGISTER(queue_create);
    REGISTER(queue_delete);

    # if 0
    REGISTER(module_load);
    REGISTER(module_unload);
    REGISTER(module_get_fn);
    # endif

    # if 0
    REGISTER(power_start);
    REGISTER(power_stop);
    # endif

    # undef REGISTER

    return (driver_t *) driver;
}

XKRT_NAMESPACE_END
