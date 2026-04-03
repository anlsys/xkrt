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

/* see https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api.html */

# define XKRT_DRIVER_ENTRYPOINT(N) XKRT_DRIVER_TYPE_ZE_ ## N

# include <xkrt/conf/conf.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver-ze.h>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger-hwloc.h>
# include <xkrt/logger/logger-ze.h>
# include <xkrt/logger/metric.h>
# include <xkrt/runtime.h>
# include <xkrt/support.h>
# include <xkrt/sync/bits.h>
# include <xkrt/sync/mutex.h>

# include <ze_api.h>
# include <hwloc/levelzero.h>
# include <hwloc/glibc-sched.h>

# include <cassert>
# include <cstdio>
# include <cstdint>
# include <cerrno>
# include <functional>

XKRT_NAMESPACE_BEGIN

// TODO : can be make this member of a 'driver_ze_t' ?  most likely yes,
// but cuda state machine would make it hard to maintain for cuda as well. Keep
// them as global variable for now, there should only be 1 instances of a
// driver right now

// xkrt
static device_ze_t * DEVICES;
static uint32_t n_devices = 0;
static bool RUNNING_ON_AURORA = 0;  // dirty fix for 2d copy segv

static device_ze_t *
device_ze_get(device_driver_id_t device_driver_id)
{
    assert(device_driver_id >= 0);
    assert((uint32_t) device_driver_id < n_devices);
    return DEVICES + device_driver_id;
}

// ze core
# define XKRT_ZE_MAX_DRIVERS 4
static ze_driver_handle_t   ze_drivers[XKRT_ZE_MAX_DRIVERS];
static uint32_t             ze_n_drivers;
static ze_context_handle_t  ze_contextes[XKRT_ZE_MAX_DRIVERS];
static ze_device_handle_t   ze_devices[XKRT_ZE_MAX_DRIVERS][XKRT_DEVICES_MAX];
static uint32_t             ze_n_devices[XKRT_ZE_MAX_DRIVERS];

// default driver to use, in case several driver exists and the API does not allow to pick one
# define ZE_DEFAULT_DRIVER_ID 0

// extensions
struct {
    ze_result_t (*zexDriverImportExternalPointer)(ze_driver_handle_t hDriver, void *ptr, size_t size);
    ze_result_t (*zexDriverReleaseImportedPointer)(ze_driver_handle_t hDriver, void *ptr);
} ext[XKRT_ZE_MAX_DRIVERS];

# if XKRT_SUPPORT_ZES
// ze sysman
static uint32_t                 zes_n_drivers;
static zes_driver_handle_t      zes_drivers[XKRT_ZE_MAX_DRIVERS];
static uint32_t                 zes_n_devices[XKRT_ZE_MAX_DRIVERS];
static zes_device_handle_t      zes_devices[XKRT_ZE_MAX_DRIVERS][XKRT_DEVICES_MAX];

// zes indexing is different than ze, this convert
static
void convert_zes_device_to_ze_device(void)
{
    // for each xkrt devices
    for (unsigned device_driver_id_t device_driver_id = 0 ; device_driver_id < n_devices ; ++device_driver_id)
    {
        // figure out the mapping ze to zes
        device_ze_t * device = device_ze_get(device_driver_id);
        uint32_t ze_device_id = device->ze.index.device;

        zes_uuid_t uuid = {};
        memcpy(uuid.id, device->ze.properties.uuid.id, ZE_MAX_DEVICE_UUID_SIZE);

        for (unsigned int zes_driver_id = 0; zes_driver_id < zes_n_drivers; ++zes_driver_id)
        {
            device->zes.device = nullptr;

            zes_device_handle_t zes_device = nullptr;
            ze_bool_t on_subdevice = false;
            uint32_t subdevice_id = -1;
            ZE_SAFE_CALL(zesDriverGetDeviceByUuidExp(zes_drivers[zes_driver_id], uuid, &zes_device, &on_subdevice, &subdevice_id));

            if (zes_device)
            {
                for (unsigned int zes_device_id = 0 ; zes_device_id < zes_n_devices[zes_driver_id] ; ++zes_device_id)
                {
                    if (zes_device == zes_devices[zes_driver_id][zes_device_id])
                    {
                        if (on_subdevice)
                            LOGGER_INFO("ZE device %2u is ZES device %u.%u", ze_device_id, zes_device_id, subdevice_id);
                        else
                            LOGGER_INFO("ZE device %2u is ZES device %2u", ze_device_id, zes_device_id);

                        device->zes.index.device = zes_device_id;
                        device->zes.index.driver = zes_driver_id;
                        device->zes.index.on_subdevice = on_subdevice;
                        device->zes.index.subdevice_id = subdevice_id;

                        device->zes.device = zes_devices[zes_driver_id][zes_device_id];

                        // memory
                        ZE_SAFE_CALL(zesDeviceEnumMemoryModules(zes_device, &device->zes.memory.count, nullptr));
                        assert(device->zes.memory.count <= XKRT_DEVICE_MEMORIES_MAX);    // if this fails, increase `XKRT_DEVICE_MEMORIES_MAX`
                        device->zes.memory.count = MIN(device->zes.memory.count, XKRT_DEVICE_MEMORIES_MAX);
                        ZE_SAFE_CALL(zesDeviceEnumMemoryModules(zes_device, &device->zes.memory.count, device->zes.memory.handles));

                        // power
                        # if 0
                        uint32_t zes_pwr_handle_count;
                        ZE_SAFE_CALL(zesDeviceEnumPowerDomains(device->zes.device, &zes_pwr_handle_count, NULL));

                        assert(zes_pwr_handle_count == 3);
                        ZE_SAFE_CALL(zesDeviceEnumPowerDomains(device->zes.device, &zes_pwr_handle_count, &device->zes.pwr.handle));
                        # endif

                        break ;
                    }
                }
                if (device->zes.device == nullptr)
                    LOGGER_FATAL("Could not map ZE device to ZES device");
            }
        }
    }
}

# endif /* XKRT_SUPPORT_ZES */

// see `ZE_FLAT_DEVICE_HIERARCHY` env variable
static int
XKRT_DRIVER_ENTRYPOINT(init)(
    unsigned int ndevices_requested,
    bool use_p2p
) {
    (void) use_p2p;

    assert(0 < ndevices_requested);
    assert(ndevices_requested <= XKRT_DEVICES_MAX);

    DEVICES = (device_ze_t *) calloc(XKRT_DEVICES_MAX, sizeof(device_ze_t));
    assert(DEVICES);

    # pragma message(TODO "We initialize all Intel drivers and devices here. Maybe make this a bit more lazy")

    // zeInit got deprecated, use other ifdef depending on version
    # if 1
    const ze_init_flags_t flags = ZE_INIT_FLAG_GPU_ONLY;
    const ze_result_t r = zeInit(flags);
    # else
    ze_init_driver_type_desc_t desc = {ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC};
    desc.pNext = nullptr;
    desc.driverType = ZE_INIT_FLAG_GPU_ONLY;
    uint32_t driverCount = 0;
    const ze_result_t r = zeInitDrivers(&driverCount, nullptr, &desc);
    # endif

    if (r != ZE_RESULT_SUCCESS)
        return 1;

    // get all drivers
    ZE_SAFE_CALL(zeDriverGet(&ze_n_drivers, NULL));
    assert(ze_n_drivers < sizeof(ze_drivers) / sizeof(*ze_drivers));
    ZE_SAFE_CALL(zeDriverGet(&ze_n_drivers, ze_drivers));

    # if XKRT_SUPPORT_ZES
    zes_init_flags_t zes_flags = ZES_INIT_FLAG_PLACEHOLDER;
    ZE_SAFE_CALL(zesInit(zes_flags));

    ZE_SAFE_CALL(zesDriverGet(&zes_n_drivers, NULL));
    assert(ze_n_drivers == zes_n_drivers);

    ZE_SAFE_CALL(zesDriverGet(&zes_n_drivers, zes_drivers));
    # endif /* XKRT_SUPPORT_ZES */

    // get all device handles per driver
    for (unsigned int ze_driver_id = 0 ; ze_driver_id < ze_n_drivers && n_devices < ndevices_requested ; ++ze_driver_id)
    {
        // get the driver
        ze_driver_handle_t ze_driver = ze_drivers[ze_driver_id];

        // get extensions - https://fossies.org/linux/mvapich/modules/libfabric/prov/psm3/psm3/psm_user.h
        ZE_SAFE_CALL(zeDriverGetExtensionFunctionAddress(ze_driver, "zexDriverImportExternalPointer", (void **) &ext[ze_driver_id].zexDriverImportExternalPointer));
        ZE_SAFE_CALL(zeDriverGetExtensionFunctionAddress(ze_driver, "zexDriverReleaseImportedPointer", (void **) &ext[ze_driver_id].zexDriverReleaseImportedPointer));

        // Create context for driver
        ze_context_desc_t ze_context_desc = {
            .stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC,
            .pNext = NULL,
            .flags = 0 // ZE_CONTEXT_FLAG_TBD
        };
        ZE_SAFE_CALL(zeContextCreate(ze_driver, &ze_context_desc, ze_contextes + ze_driver_id));

        // get devices handles
        ZE_SAFE_CALL(zeDeviceGet(ze_driver, &ze_n_devices[ze_driver_id], NULL));
        ZE_SAFE_CALL(zeDeviceGet(ze_driver, &ze_n_devices[ze_driver_id], ze_devices[ze_driver_id]));

        # if XKRT_SUPPORT_ZES
        // TODO: assuming that ze driver id == zes driver id, which is probably wrong, but fuck off honestly
        unsigned int zes_driver_id = ze_driver_id;
        zes_driver_handle_t zes_driver = zes_drivers[zes_driver_id];

        // no choice but to get all zes device handle, as mapping differs...
        ZE_SAFE_CALL(zesDeviceGet(zes_driver, &zes_n_devices[zes_driver_id], NULL));
        ZE_SAFE_CALL(zesDeviceGet(zes_driver, &zes_n_devices[zes_driver_id], zes_devices[zes_driver_id]));
        # endif /* XKRT_SUPPORT_ZES */

        // sycl interop
        # if XKRT_SUPPORT_ZE_SYCL_INTEROP
        sycl::platform platform = sycl::make_platform<sycl::backend::ext_oneapi_level_zero>(ze_driver);
        # endif /* XKRT_SUPPORT_ZE_SYCL_INTEROP */

        // create xkrt devices
        uint32_t n = MIN(ndevices_requested - n_devices, ze_n_devices[ze_driver_id]);
        for (unsigned int ze_device_id = 0 ; ze_device_id < n ; ++ze_device_id)
        {
            ze_device_handle_t ze_device = ze_devices[ze_driver_id][ze_device_id];

            // save handles
            device_ze_t * device = DEVICES + n_devices++;
            device->ze.context = ze_contextes[ze_driver_id];
            device->ze.driver  = ze_drivers[ze_driver_id];
            device->ze.handle  = ze_device;
            device->ze.index.device = ze_device_id;
            device->ze.index.driver = ze_driver_id;

            // get subdevice properties
            device->ze.properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ZE_SAFE_CALL(zeDeviceGetProperties(device->ze.handle, &device->ze.properties));

            // save device type
            device->type = DEVICE_ZE_TYPE_UNKNOWN;
            if (strstr(device->ze.properties.name, "GPU Max 1550"))
            {
                device->type = DEVICE_ZE_TYPE_AURORA25;
                RUNNING_ON_AURORA = true;
            }

            // get memory properties
            ZE_SAFE_CALL(zeDeviceGetMemoryProperties(device->ze.handle, &device->ze.memory.count, nullptr));
            assert(device->ze.memory.count <= XKRT_DEVICE_MEMORIES_MAX);    // if this fails, increase `XKRT_DEVICE_MEMORIES_MAX`
            device->ze.memory.count = MIN(device->ze.memory.count, XKRT_DEVICE_MEMORIES_MAX);
            ZE_SAFE_CALL(zeDeviceGetMemoryProperties(device->ze.handle, &device->ze.memory.count, device->ze.memory.properties));

            # if XKRT_SUPPORT_ZE_SYCL_INTEROP
            // sycl interop
            device->sycl.device = sycl::ext::oneapi::level_zero::detail::make_device(platform, (ur_native_handle_t) ze_device);

            sycl::ext::oneapi::level_zero::ownership ownership = sycl::ext::oneapi::level_zero::ownership::keep;
            sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::context> InteropContextInput{device->ze.context, std::vector<sycl::device>(1, device->sycl.device), ownership};
            device->sycl.context = sycl::make_context<sycl::backend::ext_oneapi_level_zero>(InteropContextInput);
            # endif /* XKRT_SUPPORT_ZE_SYCL_INTEROP */

            if (n_devices == ndevices_requested)
                break ;
        }
    }

    # if XKRT_SUPPORT_ZES
    convert_zes_device_to_ze_device();
    # endif /* XKRT_SUPPORT_ZES */

    return 0;
}

void
XKRT_DRIVER_ENTRYPOINT(device_info)(
    device_driver_id_t device_driver_id,
    char * buffer,
    size_t size
) {
    device_ze_t * device = device_ze_get(device_driver_id);

    char uuid[2 + 2 * ZE_MAX_DEVICE_UUID_SIZE + 1];
    size_t pos = 0;
    pos += snprintf(uuid + pos, sizeof(uuid) - pos, "0x");
    for (int i = 0 ; i < ZE_MAX_DEVICE_UUID_SIZE ; ++i)
        pos += snprintf(uuid + pos, sizeof(uuid) - pos, "%X", device->ze.properties.uuid.id[i]);

    snprintf(
        buffer,
        size,
        "Level Zero device %2d - %s with %d slices of %d subslices of %d EUs of "
        "%d threads - %.2lfGB maximum alloc - core clock rate of %.2lfGHz - "
        "timer resolution of %luns - deviceId(pci)=%d - uuid=%s",
        device_driver_id,
        device->ze.properties.name,
        device->ze.properties.numSlices,
        device->ze.properties.numSubslicesPerSlice,
        device->ze.properties.numEUsPerSubslice,
        device->ze.properties.numThreadsPerEU,
        device->ze.properties.maxMemAllocSize / 1e9,
        device->ze.properties.coreClockRate / 1e3,
        device->ze.properties.timerResolution,
        device->ze.properties.deviceId,
        uuid
    );
}

static void
XKRT_DRIVER_ENTRYPOINT(finalize)(void)
{
    // TODO : zeDeinit ?

    // get all device handles per driver
    for (int i = 0 ; i < XKRT_DEVICES_MAX && ze_contextes[i] ; ++i)
        ZE_SAFE_CALL(zeContextDestroy(ze_contextes[i]));

    free(DEVICES);
}

static const char *
XKRT_DRIVER_ENTRYPOINT(get_name)(void)
{
    return "ZE";
}

static unsigned int
XKRT_DRIVER_ENTRYPOINT(get_ndevices_max)(void)
{
    return n_devices;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_cpuset)(hwloc_topology_t topology, cpu_set_t * schedset, device_driver_id_t device_driver_id)
{
    device_ze_t * device = device_ze_get(device_driver_id);

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    HWLOC_SAFE_CALL(hwloc_levelzero_get_device_cpuset(topology, device->ze.handle, cpuset));
    CPU_ZERO(schedset);
    HWLOC_SAFE_CALL(hwloc_cpuset_to_glibc_sched_affinity(topology, cpuset, schedset, sizeof(cpu_set_t)));

    hwloc_bitmap_free(cpuset);
    return 0;
}

static device_t *
XKRT_DRIVER_ENTRYPOINT(device_create)(
    driver_t * driver,
    device_driver_id_t device_driver_id
) {
    (void) driver;
    assert(device_driver_id >= 0 && device_driver_id < XKRT_DEVICES_MAX);

    device_ze_t * device = device_ze_get(device_driver_id);
    assert(device->inherited.state == XKRT_DEVICE_STATE_DEALLOCATED);

    // query number of command queue groups
    ZE_SAFE_CALL(zeDeviceGetCommandQueueGroupProperties(device->ze.handle, &device->ze.ncommandqueuegroups, NULL));
    assert(device->ze.ncommandqueuegroups);

    // query each group
    device->ze.command_queue_group_properties = (ze_command_queue_group_properties_t *) malloc(sizeof(ze_command_queue_group_properties_t) * device->ze.ncommandqueuegroups);
    assert(device->ze.command_queue_group_properties);
    ZE_SAFE_CALL(zeDeviceGetCommandQueueGroupProperties(device->ze.handle, &device->ze.ncommandqueuegroups, device->ze.command_queue_group_properties));

    device->ze.command_queue_group_used = new std::atomic<uint32_t>[device->ze.ncommandqueuegroups];
    assert(device->ze.command_queue_group_used);
    for (uint32_t i = 0 ; i < device->ze.ncommandqueuegroups ; ++i)
        device->ze.command_queue_group_used[i].store(0);

    return (device_t *) device;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_init)(device_driver_id_t device_driver_id)
{
    // TODO : move some stuff from driver init to here
    (void) device_driver_id;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_destroy)(device_driver_id_t device_driver_id)
{
    device_ze_t * device = device_ze_get(device_driver_id);
    delete [] device->ze.command_queue_group_used;
    return 0;
}

/* Called for each device of the driver once they all have been initialized */
static int
XKRT_DRIVER_ENTRYPOINT(device_commit)(
    device_driver_id_t device_driver_id,
    device_unique_id_bitfield_t * affinity
) {
    // TODO: Intel API `zeDeviceGetP2PProperties` currently does not have a property about P2P performances
    // so instead, simply hard-code affinity for now

    device_ze_t * device = device_ze_get(device_driver_id);
    device_unique_id_t device_unique_id = device->inherited.unique_id;

    int rank = 0;
    affinity[rank++] = (1 << device_unique_id);

# if 1
    device_unique_id_t subdevice_unique_id = (device_unique_id % 2 == 0) ? (device_unique_id + 1) : (device_unique_id - 1);
    affinity[rank++] =  (1 << subdevice_unique_id);
    affinity[rank++] = (~affinity[0]) & (~affinity[1]);
# else
    LOGGER_IMPL("Set affinity, right now inter-stack link has the same affinity as xe link :-(");
    affinity[rank++] = ~affinity[0];
# endif
    assert(rank <= XKRT_DEVICES_PERF_RANK_MAX);

    return 0;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(transfer_async)(
    void * dst,
    void * src,
    const size_t size,
    command_queue_t * iqueue,
    ze_event_handle_t ze_event_handle = nullptr
) {
    queue_ze_t * queue = (queue_ze_t *) iqueue;
    assert(queue);

    const uint32_t num_wait_events = 0;
    ze_event_handle_t * wait_events = NULL;

    ZE_SAFE_CALL(
        zeCommandListAppendMemoryCopy(
            queue->ze.command.list,
            dst,
            src,
            size,
            ze_event_handle,
            num_wait_events,
            wait_events
        )
    );
    return 0;
}

////////////
// QUEUE //
////////////

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(
    command_queue_t * iqueue
) {
    queue_ze_t * queue = (queue_ze_t *) iqueue;
    assert(queue);

    const uint64_t timeout = UINT64_MAX;
# if 1
    ZE_SAFE_CALL(zeCommandListHostSynchronize(queue->ze.command.list, timeout));
# else
    LOGGER_FATAL("Not supported");
# endif
    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_launch)(
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    queue_ze_t * queue = (queue_ze_t *) iqueue;
    assert(queue);

    ze_event_handle_t ze_event_handle = queue->ze.events.list[idx];

    // TODO : try zeCommandListAppendEventReset and see if it reduces latency
    ZE_SAFE_CALL(zeEventHostReset(ze_event_handle));

    const uint32_t num_wait_events = 0;
    ze_event_handle_t * wait_events = NULL;

    int err = EINPROGRESS;

    switch (cmd->type)
    {
        case (ocg::COMMAND_TYPE_PROG):
        {
            LOGGER_FATAL("IMPL ME");
            break ;
        }

        case (ocg::COMMAND_TYPE_COPY_H2D_1D):
        case (ocg::COMMAND_TYPE_COPY_D2H_1D):
        case (ocg::COMMAND_TYPE_COPY_D2D_1D):
        {
                  void * dst    = (      void *) cmd->copy_1D.dst_device_addr;
            const void * src    = (const void *) cmd->copy_1D.src_device_addr;
            const size_t size   = cmd->copy_1D.size;
            ZE_SAFE_CALL(
                zeCommandListAppendMemoryCopy(
                    queue->ze.command.list,
                    dst,
                    src,
                    size,
                    ze_event_handle,
                    num_wait_events,
                    wait_events
                )
            );
            break ;
        }

        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
        {
                  void * dst    = (      void *) cmd->copy_2D.dst_device_view.addr;
            const void * src    = (const void *) cmd->copy_2D.src_device_view.addr;

            const size_t dst_pitch = cmd->copy_2D.dst_device_view.ld * cmd->copy_2D.sizeof_type;
            const size_t src_pitch = cmd->copy_2D.src_device_view.ld * cmd->copy_2D.sizeof_type;

            const size_t width  = cmd->copy_2D.m * cmd->copy_2D.sizeof_type;
            const size_t height = cmd->copy_2D.n;

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

            // aurora dirty fix
            if (RUNNING_ON_AURORA &&
                   (src_region.width != src_pitch || dst_region.width != dst_pitch))
            {
                LOGGER_ERROR("memcpy2D not working on intel max gpu series zzzzzzzzzz.... serializing synchronously line by line using calling host thread");
                assert(src_region.height == dst_region.height);
                for (int i = 0 ; i < src_region.height ; ++i)
                {
                    void * l_dst = (void *) (cmd->copy_2D.dst_device_view.addr + i*dst_pitch);
                    void * l_src = (void *) (cmd->copy_2D.src_device_view.addr + i*src_pitch);
                    const size_t l_size = width;
                    const ze_event_handle_t l_ze_event_handle = (i == src_region.height - 1) ? ze_event_handle : nullptr;
                    XKRT_DRIVER_ENTRYPOINT(transfer_async)(l_dst, l_src, l_size, iqueue, l_ze_event_handle);
                }
                XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(iqueue);
            }
            else
            {
                ZE_SAFE_CALL(
                    zeCommandListAppendMemoryCopyRegion(
                        queue->ze.command.list,
                        dst,
                       &dst_region,
                        dst_pitch,
                        dst_slice_pitch,
                        src,
                       &src_region,
                        src_pitch,
                        src_slice_pitch,
                        ze_event_handle,
                        num_wait_events,
                        wait_events
                    )
                );
            }
            break ;
        }

        default:
            return EINVAL;
    }

    return err;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait)(
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    queue_ze_t * queue = (queue_ze_t *) iqueue;
    assert(queue);

    ze_event_handle_t event = queue->ze.events.list[idx];
    const uint64_t timeout = UINT64_MAX;

    ZE_SAFE_CALL(zeEventHostSynchronize(event, timeout));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_suggest)(
    device_driver_id_t device_driver_id,
    command_queue_type_t qtype
) {
    (void) device_driver_id;

    switch (qtype)
    {
        case (XKRT_QUEUE_TYPE_KERN):
            return 8;

        case (XKRT_QUEUE_TYPE_H2D):
        case (XKRT_QUEUE_TYPE_D2H):
        case (XKRT_QUEUE_TYPE_D2D):
            return 4;

        case (XKRT_QUEUE_TYPE_FD_READ):
        case (XKRT_QUEUE_TYPE_FD_WRITE):
            return 0;

        default:
            return 1;
    }
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_progress)(
    command_queue_t * iqueue
) {
    assert(iqueue);

    queue_ze_t * queue = (queue_ze_t *) iqueue;
    int r = 0;

    iqueue->progress([&] (command_t * cmd, xkrt_command_queue_list_counter_t p) {

        switch (cmd->type)
        {
            case (ocg::COMMAND_TYPE_PROG):
            case (ocg::COMMAND_TYPE_COPY_H2H_1D):
            case (ocg::COMMAND_TYPE_COPY_H2D_1D):
            case (ocg::COMMAND_TYPE_COPY_D2H_1D):
            case (ocg::COMMAND_TYPE_COPY_D2D_1D):
            case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            case (ocg::COMMAND_TYPE_COPY_D2D_2D):
            {
                ze_event_handle_t event = queue->ze.events.list[p];
                ze_result_t res = zeEventQueryStatus(event);
                if (res == ZE_RESULT_NOT_READY)
                    r = EINPROGRESS;
                else if (res == ZE_RESULT_SUCCESS)
                    iqueue->complete_command(p);
                else
                    ZE_SAFE_CALL(res);

                break ;
            }

            default:
                LOGGER_FATAL("Wrong command");
        }

        return true;
    });

    return r;
}

template<typename T>
static inline bool
f_equals(
    const T & x,
    const T & y
) {
    return x == y ? true : false;
}

template<typename T>
static inline bool
f_and(
    const T & x,
    const T & y
) {
    return x & y ? true : false;
}

// return the next command queue group to use
template <typename T, bool (*f)(const T & x, const T & y)>
static inline uint32_t
device_command_queue_group_next(
    device_ze_t * device,
    const ze_command_queue_group_property_flag_t & flag
) {
    uint32_t ordinal_with_least_queues = UINT32_MAX;
    uint32_t min_queues = UINT32_MAX;

    for (uint32_t i = 0; i < device->ze.ncommandqueuegroups; ++i)
    {
        ze_command_queue_group_properties_t * properties = device->ze.command_queue_group_properties + i;
        if (f((const ze_command_queue_group_property_flag_t &) properties->flags, flag))
        {
            const uint32_t used = device->ze.command_queue_group_used[i].load();
            if (used < min_queues)
            {
                min_queues = used;
                ordinal_with_least_queues = i;
            }
        }
    }

    return ordinal_with_least_queues;
}

static command_queue_t *
XKRT_DRIVER_ENTRYPOINT(command_queue_create)(
    device_t * idevice,
    command_queue_type_t type,
    xkrt_command_queue_list_counter_t capacity
) {
    assert(idevice);

    queue_ze_t * queue = (queue_ze_t *) malloc(sizeof(queue_ze_t));
    assert(queue);

    command_queue_init(
        (command_queue_t *) queue,
        type,
        capacity,
        XKRT_DRIVER_ENTRYPOINT(command_queue_launch),
        XKRT_DRIVER_ENTRYPOINT(command_queue_progress),
        XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all),
        XKRT_DRIVER_ENTRYPOINT(command_queue_wait)
    );

    device_ze_t * device = (device_ze_t *) idevice;
    queue->ze.device = device;

    // Round robin over copy engines

    # if 0
    // convert xkrt queue type to a command queue group flag
    ze_command_queue_group_property_flag_t flag;
    switch (type)
    {
        case (XKRT_QUEUE_TYPE_H2D):
        case (XKRT_QUEUE_TYPE_D2H):
        case (XKRT_QUEUE_TYPE_D2D):
        {
            flag = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY;
            break ;
        }

        case (XKRT_QUEUE_TYPE_KERN):
        {
            flag = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
            break ;
        }

        default:
            LOGGER_FATAL("Unknown queue type");
    }

    uint32_t ordinal = device_command_queue_group_next<ze_command_queue_group_property_flag_t, f_equals>(device, flag);
    if (ordinal == UINT32_MAX)
    {
        ordinal = device_command_queue_group_next<ze_command_queue_group_property_flag_t, f_and>(device, flag);
        if (ordinal == UINT32_MAX)
            LOGGER_FATAL("No command queue group available for queue");
    }
    # else
    /* https://github.com/pmodels/mpich/blob/main/src/mpl/src/gpu/mpl_gpu_ze.c#L656-L660 */
    uint32_t ordinal;
    switch (type)
    {
        case (XKRT_QUEUE_TYPE_KERN):
        {
            ordinal = 0;
            break ;
        }

        case (XKRT_QUEUE_TYPE_H2D):
        case (XKRT_QUEUE_TYPE_D2H):
        case (XKRT_QUEUE_TYPE_D2D):
        {
            ordinal = 1;
            break ;
        }

        default:
            LOGGER_FATAL("Unknown queue type");
    }
    # endif

    // retrieve group properties
    const ze_command_queue_group_properties_t * properties = device->ze.command_queue_group_properties + ordinal;
    uint32_t index = device->ze.command_queue_group_used[ordinal].fetch_add(1) % properties->numQueues;

    // get the next command queue index to use in the group
    const ze_command_queue_desc_t ze_command_queue_desc = {
        .stype      = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC,
        .pNext      = NULL,
        .ordinal    = ordinal,
        .index      = index,
        .flags      = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY,
        .mode       = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS,
        .priority   = ZE_COMMAND_QUEUE_PRIORITY_NORMAL // ZE_COMMAND_QUEUE_PRIORITY_PRIORITY_LOW
    };
    LOGGER_DEBUG("Creating queue of type `%4s` with (ordinal, index) = (%d, %d)", ocg::command_type_to_str((ocg::command_type_t)type), ordinal, index);

    # if 0 /* use a command list and command queue */
    ZE_SAFE_CALL(
        zeCommandQueueCreate(
            device->ze.context,
            device->ze.handle,
           &ze_command_queue_desc,
           &queue->ze.command.queue
        )
    );

    // create command list
    ze_command_list_desc_t ze_command_list_desc = {
        .stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC,
        .pNext = NULL,
        .commandQueueGroupOrdinal = ordinal,
        .flags = ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING | ZE_COMMAND_LIST_FLAG_MAXIMIZE_THROUGHPUT
    };
    # else /* use command list immediate */
    ZE_SAFE_CALL(
        zeCommandListCreateImmediate(
            device->ze.context,
            device->ze.handle,
           &ze_command_queue_desc,
           &queue->ze.command.list
        )
    );

    # if XKRT_SUPPORT_ZE_SYCL_INTEROP
    sycl::ext::oneapi::level_zero::ownership ownership = sycl::ext::oneapi::level_zero::ownership::keep;
    sycl::backend_input_t<sycl::backend::ext_oneapi_level_zero, sycl::queue> InteropQueueInput{queue->ze.command.list, device->sycl.device, ownership};
    sycl::queue sycl_queue = sycl::make_queue<sycl::backend::ext_oneapi_level_zero>(
        InteropQueueInput,
        device->sycl.context
    );

    new (&queue->sycl.queue) sycl::queue(sycl_queue);
    # endif /* XKRT_SUPPORT_ZE_SYCL_INTEROP */

    # endif

    // create event pool and events
    const ze_event_pool_desc_t ze_event_pool_desc = {
        .stype  = ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        .pNext  = NULL,
        .flags  = ZE_EVENT_POOL_FLAG_HOST_VISIBLE,
        .count  = capacity
    };
    const uint32_t ndevices = 1;
    ZE_SAFE_CALL(zeEventPoolCreate(device->ze.context, &ze_event_pool_desc, ndevices, &device->ze.handle, &queue->ze.events.pool));

    queue->ze.events.list = (ze_event_handle_t *) malloc(sizeof(ze_event_handle_t) * capacity);
    assert(queue->ze.events.list);
    for (xkrt_command_queue_list_counter_t i = 0 ; i < capacity ; ++i)
    {
        ze_event_desc_t event_desc = {
            .stype  = ZE_STRUCTURE_TYPE_EVENT_DESC,
            .pNext  = NULL,
            .index  = (uint32_t) i,
            .signal = ZE_EVENT_SCOPE_FLAG_HOST,
            .wait   = ZE_EVENT_SCOPE_FLAG_HOST
        };
        ZE_SAFE_CALL(zeEventCreate(queue->ze.events.pool, &event_desc, queue->ze.events.list + i));
    }

    return (command_queue_t *) queue;
}

static void
XKRT_DRIVER_ENTRYPOINT(command_queue_delete)(
    command_queue_t * iqueue
) {
    queue_ze_t * queue = (queue_ze_t *) iqueue;
    ZE_SAFE_CALL(zeEventPoolDestroy(queue->ze.events.pool));
    ZE_SAFE_CALL(zeCommandListDestroy(queue->ze.command.list));
    free(queue);
}

////////////
// MEMORY //
////////////

static void *
XKRT_DRIVER_ENTRYPOINT(memory_device_allocate)(device_driver_id_t device_driver_id, const size_t size, int area_idx)
{
    if (size == 0)
        return NULL;

    device_ze_t * device = device_ze_get(device_driver_id);

    # if 1
    const ze_device_mem_alloc_desc_t device_desc = {
        .stype = ZE_STRUCTURE_TYPE_DEVICE_MEMORY_PROPERTIES,
        .pNext = NULL,
        .flags = 0,
        .ordinal = (uint32_t) area_idx // device memory ordinal (should be here where we see HBM/DRAM)
    };
    const size_t alignment = 4 * sizeof(double);
    void * device_ptr = NULL;
    ze_result_t res = zeMemAllocDevice(device->ze.context, &device_desc, size, alignment, device->ze.handle, &device_ptr);
    # else

    // TODO : cannot select memory ordinal with virtual/physical memory API

    // Query page size for our allocation
    size_t pagesize;
    ZE_SAFE_CALL(
        zeVirtualMemQueryPageSize(
            device->ze.context,
            device->ze.handle,
            size,
           &pagesize
        )
    );

    // Align size and reserve virtual address space.
    const size_t reserve_size = size + (pagesize - (size % pagesize));
    void * device_ptr = NULL;
    ZE_SAFE_CALL(
        zeVirtualMemReserve(
            device->ze.context,
            NULL,
            reserve_size,
           &device_ptr
        )
    );
    assert(device_ptr);

    // Create physical memory
    ze_physical_mem_desc_t ze_physical_mem_desc = {
        .stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC,
        .pNext = NULL,
        .flags = ZE_PHYSICAL_MEM_FLAG_TBD,
        .size  = reserve_size
    };
    ze_physical_mem_handle_t ze_physical_mem_handle;
    ZE_SAFE_CALL(
        zePhysicalMemCreate(
            device->ze.context,
            device->ze.handle,
           &ze_physical_mem_desc,
           &ze_physical_mem_handle
        )
    );

    // Map virtual to physical memory
    const ze_memory_access_attribute_t ze_memory_access_attribute = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
    const size_t offset = 0;
    ZE_SAFE_CALL(
        zeVirtualMemMap(
            device->ze.context,
            device_ptr,
            size,
            ze_physical_mem_handle,
            offset,
            ze_memory_access_attribute
        )
    );
    # endif

    if (res == ZE_RESULT_SUCCESS)
    {
        res = zeContextMakeMemoryResident(device->ze.context, device->ze.handle, device_ptr, size);
        if (res == ZE_RESULT_SUCCESS)
            return device_ptr;

        ZE_SAFE_CALL(zeMemFree(device->ze.context, device_ptr));
    }

    return NULL;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_deallocate)(device_driver_id_t device_driver_id, void * ptr, const size_t size, int area_idx)
{
    (void) size;
    (void) area_idx;
    if (ptr)
    {
        device_ze_t * device = device_ze_get(device_driver_id);

        // from level zero spec:
        //  The application may free the memory without evicting; the memory is implicitly evicted when freed.
        ZE_SAFE_CALL(zeMemFree(device->ze.context, ptr));
    }
}

// TODO
//
// WARNING 1
//  If built with `XKRT_SUPPORT_ZES` - then 2 memory areas will be reported (the HBM of each tile) - whatever the `ZE_FLAT_DEVICE_HIERARCHY`
//  Otherwise, it will report 1 memory of 128GB virtualizing the 2 HBM with `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` - or 1 memory of 64GB if `ZE_FLAT_DEVICE_HIERARCHY=FLAT` that is the stack HBM
static void
XKRT_DRIVER_ENTRYPOINT(memory_device_info)(
    device_driver_id_t device_driver_id,
    device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX],
    int * nmemories
) {
    device_ze_t * device = device_ze_get(device_driver_id);

    # if XKRT_SUPPORT_ZES

    // TODO: how to get memory mapping to subdevice ? we currently cannot, so
    // hardcode 1 memory per device and assume sysman reports in subdevice
    // index order...
    unsigned int zes_memory_offset;
    if (device->zes.index.on_subdevice)
    {
        *nmemories = 1;
        zes_memory_offset = device->zes.index.subdevice_id;
        // *nmemories = device->zes.memory.count;
        // zes_memory_offset = 0;
    }
    else
    {
        *nmemories = device->zes.memory.count;
        zes_memory_offset = 0;
    }

    for (uint32_t i = 0 ; i < *nmemories && i < XKRT_DEVICE_MEMORIES_MAX ; ++i)
    {
        // TODO: how to get memory name with zes ? because most likely ze
        // memory mapping is different from zes memory mapping...
        strncpy(info[i].name, "(null)", sizeof(info[i].name));

        assert(i + zes_memory_offset < device->zes.memory.count);
        zes_mem_handle_t memory = device->zes.memory.handles[i + zes_memory_offset];
        zes_mem_state_t state = {
            .stype = ZES_STRUCTURE_TYPE_MEM_STATE,
            .pNext = NULL,
            .health = ZES_MEM_HEALTH_UNKNOWN,
            .free = 0,
            .size = 0,
        };
        ZE_SAFE_CALL(zesMemoryGetState(memory, &state));
        info[i].used = state.size - state.free;
        info[i].capacity = state.size;
    }

    # else

    *nmemories = device->ze.memory.count;
    for (uint32_t i = 0 ; i < *nmemories && i < XKRT_DEVICE_MEMORIES_MAX ; ++i)
    {
        strncpy(info[i].name, device->ze.memory.properties[i].name, MIN(sizeof(device->ze.memory.properties[i].name), sizeof(info[i].name)));
        info[i].used = SIZE_MAX;
        info[i].capacity = device->ze.memory.properties[i].totalSize;
    }

    # endif /* XKRT_SUPPORT_ZES */
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_host_allocate)(
    device_driver_id_t device_driver_id,
    uint64_t size
) {
    device_ze_t * device = device_ze_get(device_driver_id);
    const ze_host_mem_alloc_desc_t host_desc = {
        .stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC,
        .pNext = NULL,
        .flags = 0
        // .flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT
        // .flags = ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED | ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT | ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED
    };
    constexpr size_t alignment = 0;
    void * ptr;
    ZE_SAFE_CALL(zeMemAllocHost(device->ze.context, &host_desc, size, alignment, (void **) &ptr));
    return ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_host_deallocate)(
    device_driver_id_t device_driver_id,
    void * mem,
    uint64_t size
) {
    (void) size;

    device_ze_t * device = device_ze_get(device_driver_id);
    ZE_SAFE_CALL(zeMemFree(device->ze.context, mem));
}

driver_module_t
XKRT_DRIVER_ENTRYPOINT(module_load)(
    device_driver_id_t device_driver_id,
    uint8_t * bin,
    size_t binsize,
    driver_module_format_t format
) {
    ze_module_format_t ze_format;
    switch (format)
    {
        case (XKRT_DRIVER_MODULE_FORMAT_SPIRV):
        {
            ze_format = ZE_MODULE_FORMAT_IL_SPIRV;
            break ;
        }

        case (XKRT_DRIVER_MODULE_FORMAT_NATIVE):
        {
            ze_format = ZE_MODULE_FORMAT_NATIVE;
            break ;
        }

        default:
            LOGGER_FATAL("Unknown format");
    }
    device_ze_t * device = device_ze_get(device_driver_id);
    ze_module_desc_t desc = {
        .stype = ZE_STRUCTURE_TYPE_MODULE_DESC,
        .pNext = NULL,
        .format = ze_format,
        .inputSize = binsize,
        .pInputModule = bin,
        .pBuildFlags = NULL,
        .pConstants = NULL
    };

    // TODO : build log
    driver_module_t module = NULL;
    ZE_SAFE_CALL(zeModuleCreate(device->ze.context, device->ze.handle, &desc, (ze_module_handle_t *) &module, NULL));
    assert(module);

    return module;
}

# if XKRT_SUPPORT_ZES

void
XKRT_DRIVER_ENTRYPOINT(power_start)(device_driver_id_t device_driver_id, power_t * pwr)
{
    // problem is, there is multiple power handle for a ze device: which one to use ?
    LOGGER_FATAL("Not implemented");

    device_ze_t * device = device_ze_get(device_driver_id);
    assert(device);
    assert(pwr);

    pwr->priv.t1 = get_nanotime();
    zes_power_energy_counter_t * e1 = (zes_power_energy_counter_t *) &pwr->priv.c1;
    ZE_SAFE_CALL(zesPowerGetEnergyCounter(device->zes.pwr.handle, e1));

    // if this fails, increase sizeof pwr->priv.c1
    static_assert(sizeof(zes_power_energy_counter_t) <= sizeof(pwr->priv.c1));
}

void
XKRT_DRIVER_ENTRYPOINT(power_stop)(device_driver_id_t device_driver_id, power_t * pwr)
{
    // problem is, there is multiple power handle for a ze device: which one to use ?
    LOGGER_FATAL("Not implemented");

    device_ze_t * device = device_ze_get(device_driver_id);
    assert(device);

    zes_power_energy_counter_t * e1 = (zes_power_energy_counter_t *) &pwr->priv.c1;
    zes_power_energy_counter_t * e2 = (zes_power_energy_counter_t *) &pwr->priv.c2;

    ZE_SAFE_CALL(zesPowerGetEnergyCounter(device->zes.pwr.handle, e2));
    pwr->priv.t2 = get_nanotime();

    double uJ = (double) (e2->energy - e1->energy);
    double  J = uJ / (double)1e6;
    double  s = (pwr->priv.t2 - pwr->priv.t1) / (double) 1e9;
    pwr->dt = s;
    pwr->P  = J / s;
}

# endif /* XKRT_SUPPORT_ZES */

void
XKRT_DRIVER_ENTRYPOINT(module_unload)(
    driver_module_t module
) {
    ZE_SAFE_CALL(zeModuleDestroy((ze_module_handle_t) module));
}

driver_module_fn_t
XKRT_DRIVER_ENTRYPOINT(module_get_fn)(
    driver_module_t module,
    const char * name
) {
    driver_module_fn_t fn = NULL;
    ze_kernel_desc_t desc = {
        .stype = ZE_STRUCTURE_TYPE_KERNEL_DESC,
        .pNext = NULL,
        .flags = ZE_KERNEL_FLAG_FORCE_RESIDENCY,
        .pKernelName = name
    };
    ZE_SAFE_CALL(zeKernelCreate((ze_module_handle_t) module, &desc, (ze_kernel_handle_t *) &fn));
    assert(fn);
    return fn;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_h2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    return XKRT_DRIVER_ENTRYPOINT(transfer_async)(dst, src, size, iqueue);
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2h_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    return XKRT_DRIVER_ENTRYPOINT(transfer_async)(dst, src, size, iqueue);
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    return XKRT_DRIVER_ENTRYPOINT(transfer_async)(dst, src, size, iqueue);
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_register)(
    void * ptr,
    uint64_t size
) {
    if (ext[ZE_DEFAULT_DRIVER_ID].zexDriverImportExternalPointer == NULL)
        LOGGER_FATAL("zexDriverImportExternalPointer is NULL");
    ZE_SAFE_CALL(ext[ZE_DEFAULT_DRIVER_ID].zexDriverImportExternalPointer(ze_drivers[ZE_DEFAULT_DRIVER_ID], ptr, size));
    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_unregister)(
    void * ptr,
    uint64_t size
) {
    (void) size;
    if (ext[ZE_DEFAULT_DRIVER_ID].zexDriverReleaseImportedPointer == NULL)
        LOGGER_FATAL("zexDriverReleaseImportedPointer is NULL");
    ZE_SAFE_CALL(ext[ZE_DEFAULT_DRIVER_ID].zexDriverReleaseImportedPointer(ze_drivers[ZE_DEFAULT_DRIVER_ID], ptr));

    (void) size;

    return 0;
}

//////////////////////////
// Routine registration //
//////////////////////////
driver_t *
XKRT_DRIVER_ENTRYPOINT(create_driver)(void)
{
    driver_ze_t * driver = (driver_ze_t *) calloc(1, sizeof(driver_ze_t));
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
    REGISTER(device_cpuset);
    REGISTER(device_info);

    # if 0
    REGISTER(transfer_h2d);
    REGISTER(transfer_d2h);
    REGISTER(transfer_d2d);
    # endif
    REGISTER(transfer_h2d_async);
    REGISTER(transfer_d2h_async);
    REGISTER(transfer_d2d_async);

    REGISTER(memory_device_info);
    REGISTER(memory_device_allocate);
    REGISTER(memory_device_deallocate);
    REGISTER(memory_host_allocate);
    REGISTER(memory_host_deallocate);
    REGISTER(memory_host_register);
    REGISTER(memory_host_unregister);
    // REGISTER(memory_unified_allocate);
    // REGISTER(memory_unified_deallocate);

    REGISTER(command_queue_suggest);
    REGISTER(command_queue_create);
    REGISTER(command_queue_delete);

    REGISTER(module_load);
    REGISTER(module_unload);
    REGISTER(module_get_fn);

    # if XKRT_SUPPORT_ZES
    REGISTER(power_start);
    REGISTER(power_stop);
    # endif /* XKRT_SUPPORT_ZES */

    # undef REGISTER

    return (driver_t *) driver;
}

XKRT_NAMESPACE_END
