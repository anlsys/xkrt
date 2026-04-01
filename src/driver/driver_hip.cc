/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
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

# define XKRT_DRIVER_ENTRYPOINT(N) XKRT_DRIVER_TYPE_HIP_ ## N

# include <xkrt/runtime.h>
# include <xkrt/support.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/driver-hip.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/logger-hip.h>
# include <xkrt/logger/logger-hipblas.h>
# include <xkrt/logger/logger-hwloc.h>
# include <xkrt/logger/metric.h>
# include <xkrt/sync/bits.h>
# include <xkrt/sync/mutex.h>

# include <hip/hip_runtime.h>
# include <rocm_smi/rocm_smi.h>
# include <hipblas/hipblas.h>

# include <hwloc.h>
# include <hwloc/rsmi.h>
# include <hwloc/glibc-sched.h>

# include <cassert>
# include <cstdio>
# include <cstdint>
# include <cerrno>

XKRT_NAMESPACE_BEGIN

/* number of used device for this run */
static device_hip_t DEVICES[XKRT_DEVICES_MAX];

static inline device_t *
device_get(device_driver_id_t device_driver_id)
{
    return (device_t *) (DEVICES + device_driver_id);
}

static inline device_hip_t *
device_hip_get(device_driver_id_t device_driver_id)
{
    return (device_hip_t *) device_get(device_driver_id);
}

static inline void
hip_set_context(device_driver_id_t device_driver_id)
{
    device_hip_t * device = device_hip_get(device_driver_id);
    HIP_SAFE_CALL(hipSetDevice(device_driver_id));
}

static unsigned int
XKRT_DRIVER_ENTRYPOINT(get_ndevices_max)(void)
{
    int device_count = 0;
    HIP_SAFE_CALL(hipGetDeviceCount(&device_count));
    return (unsigned int)device_count;
}

/* hip_perf_topo[i,j] returns the perf_rank of the communication link between
   device.
   hip_perf_device[d][i] for i=0,..,XKRT_DEVICES_PERF_RANK_MAX-1 is the mask of device
   for which the device d has link with performance i.
*/

static int                                hip_device_count   = 0;
static int                              * hip_perf_topo      = NULL;
static device_unique_id_bitfield_t * hip_perf_device    = NULL;
static bool                               hip_use_p2p        = false;

static void
get_gpu_topo(
    unsigned int ndevices,
    bool use_p2p
) {
    hip_device_count = ndevices;

    hip_perf_topo = (int *) malloc(sizeof(int) * hip_device_count * hip_device_count);
    assert(hip_perf_topo);

    int rank_used[64];
    memset(rank_used, 0, sizeof(rank_used));
    rank_used[0] = 1;

    // Enumerates Device <-> Device links and store perf_rank
    for (int i = 0; i < hip_device_count; ++i)
    {
        device_hip_t * device_i = device_hip_get(i);
        for (int j = 0; j < hip_device_count; ++j)
        {
            const int idx = i*hip_device_count+j;
            if (i == j)
            {
                // device to same device = highest perf
                hip_perf_topo[idx] = 0;
            }
            else
            {
                hip_perf_topo[i*hip_device_count+j] = INT_MAX;

                if (use_p2p)
                {
                    device_hip_t * device_j = device_hip_get(j);
                    int perf_rank = 0;
                    int access_supported = 0;

                    HIP_SAFE_CALL(
                            hipDeviceGetP2PAttribute(
                                &access_supported,
                                hipDevP2PAttrAccessSupported,
                                device_i->hip.device,
                                device_j->hip.device
                                )
                            );

                    if (access_supported)
                    {
                        HIP_SAFE_CALL(
                                hipDeviceGetP2PAttribute(
                                    &perf_rank,
                                    hipDevP2PAttrPerformanceRank,
                                    device_i->hip.device,
                                    device_j->hip.device
                                    )
                                );

                        hip_perf_topo[i*hip_device_count+j] = 1 + perf_rank;

                        if (1 + perf_rank >= sizeof(rank_used) / sizeof(*rank_used))
                            LOGGER_FATAL("P2P perf_rank too high");
                        rank_used[1 + perf_rank] = 1;
                    }
                }
            }
        }
    }

    /* shrink perf ranks, on MI300A it starts at 4 somehow */
    for (int i = 0 ; i < hip_device_count*hip_device_count ; ++i)
    {
        if (hip_perf_topo[i] == INT_MAX)
            hip_perf_topo[i] = XKRT_DEVICES_PERF_RANK_MAX - 1;
        else
        {
            const int perf_rank = hip_perf_topo[i];
            int rank = perf_rank;
            while (rank - 1 > 0 && rank_used[rank - 1] == 0)
                --rank;

            if (rank != perf_rank)
            {
                for (int j = i ; j < hip_device_count*hip_device_count ; ++j)
                    if (hip_perf_topo[j] == perf_rank)
                        hip_perf_topo[j] = rank;

                rank_used[rank]      = 1;
                rank_used[perf_rank] = 0;
            }
        }

        if (hip_perf_topo[i] >= XKRT_DEVICES_PERF_RANK_MAX)
            LOGGER_FATAL("Too many perf ranks. Recompile increasing `XKRT_DEVICES_PERF_RANK_MAX` to at least %d", XKRT_DEVICES_PERF_RANK_MAX);
    }

    // get number of ranks
    size_t size = hip_device_count * XKRT_DEVICES_PERF_RANK_MAX * sizeof(device_unique_id_bitfield_t);
    hip_perf_device = (device_unique_id_bitfield_t *) malloc(size);
    assert(hip_perf_device);
    memset(hip_perf_device, 0, size);

    for (int device_hip_id = 0 ; device_hip_id < hip_device_count ; ++device_hip_id)
    {
        for (int other_device_hip_id = 0 ; other_device_hip_id < hip_device_count ; ++other_device_hip_id)
        {
            int rank = hip_perf_topo[device_hip_id*hip_device_count+other_device_hip_id];
            assert(0 <= device_hip_id * hip_device_count   + rank);
            assert(     device_hip_id * XKRT_DEVICES_PERF_RANK_MAX + rank <= hip_device_count * XKRT_DEVICES_PERF_RANK_MAX);

            hip_perf_device[device_hip_id * XKRT_DEVICES_PERF_RANK_MAX + rank] |= (1 << other_device_hip_id);
        }
    }
}

static int
XKRT_DRIVER_ENTRYPOINT(init)(
    unsigned int ndevices,
    bool use_p2p
) {
    hipError_t err = hipInit(0);
    if (err != hipSuccess)
        return 1;

    rsmi_status_t rs = rsmi_init(0);
    if (rs != RSMI_STATUS_SUCCESS)
        return 1;

    hip_use_p2p = use_p2p;

    int ndevices_max;
    err = hipGetDeviceCount(&ndevices_max);
    if (err)
        return 1;
    ndevices = MIN((int)ndevices, ndevices_max);

    // TODO : move that to device init
    assert(ndevices <= XKRT_DEVICES_MAX);
    for (unsigned int i = 0 ; i < ndevices ; ++i)
    {
        device_hip_t * device = device_hip_get(i);
        device->inherited.state = XKRT_DEVICE_STATE_DEALLOCATED;
        HIP_SAFE_CALL(hipDeviceGet(&device->hip.device, i));
    }

    get_gpu_topo(ndevices, use_p2p);
    return 0;
}

static void
XKRT_DRIVER_ENTRYPOINT(finalize)(void)
{
    rsmi_status_t  rs = rsmi_shut_down();
    if (rs != RSMI_STATUS_SUCCESS)
        LOGGER_FATAL("`rsmi_shut_down` failed");

}

static const char *
XKRT_DRIVER_ENTRYPOINT(get_name)(void)
{
    return "HIP";
}

static int
XKRT_DRIVER_ENTRYPOINT(device_cpuset)(
    hwloc_topology_t topology,
    cpu_set_t * schedset,
    device_driver_id_t device_driver_id
) {
    assert(device_driver_id >= 0);
    assert(device_driver_id < XKRT_DEVICES_MAX);

    hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
    HWLOC_SAFE_CALL(hwloc_rsmi_get_device_cpuset(topology, device_driver_id, cpuset));

    CPU_ZERO(schedset);
    HWLOC_SAFE_CALL(hwloc_cpuset_to_glibc_sched_affinity(topology, cpuset, schedset, sizeof(cpu_set_t)));

    hwloc_bitmap_free(cpuset);

    return 0;
}

static device_t *
XKRT_DRIVER_ENTRYPOINT(device_create)(driver_t * driver, device_driver_id_t device_driver_id)
{
    (void) driver;

    assert(device_driver_id >= 0 && device_driver_id < XKRT_DEVICES_MAX);

    device_hip_t * device = device_hip_get(device_driver_id);
    assert(device->inherited.state == XKRT_DEVICE_STATE_DEALLOCATED);

    return (device_t *) device;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_init)(device_driver_id_t device_driver_id)
{
    hip_set_context(device_driver_id);

    device_hip_t * device = device_hip_get(device_driver_id);
    assert(device);
    assert(device->inherited.state == XKRT_DEVICE_STATE_CREATE);

    HIP_SAFE_CALL(hipDeviceGetAttribute(&device->hip.prop.pciBusID,    hipDeviceAttributePciBusId,         device->hip.device));
    HIP_SAFE_CALL(hipDeviceGetAttribute(&device->hip.prop.pciDeviceID, hipDeviceAttributePciDeviceId,      device->hip.device));

    memset(device->hip.prop.name, 0, sizeof(device->hip.prop.name));
    HIP_SAFE_CALL(hipDeviceGetName(device->hip.prop.name, sizeof(device->hip.prop.name), device->hip.device));

    HIP_SAFE_CALL(hipDeviceTotalMem(&device->hip.prop.mem_total, device->hip.device));
}

# define USE_MMAP_EXPLICITLY 0

# if USE_MMAP_EXPLICITLY
static inline void
get_prop_and_size(
    device_driver_id_t device_driver_id,
    const size_t size,
    hipMemAllocationProp * prop,
    size_t * actualsize
) {
    prop->type = hipMemAllocationTypePinned;
    prop->requestedHandleTypes = hipMemHandleTypeNone;
    prop->location.type = hipMemLocationTypeDevice;
    prop->location.id = device_driver_id;
    prop->win32HandleMetaData = NULL;
    prop->allocFlags.compressionType = hipMemaccess_tFlagsProtNone;
    prop->allocFlags.gpuDirectRDMACapable = 0;
    prop->allocFlags.usage = 0;
    prop->allocFlags.reserved[0] = 0;
    prop->allocFlags.reserved[1] = 0;
    prop->allocFlags.reserved[2] = 0;
    prop->allocFlags.reserved[3] = 0;

    size_t granularity;
    HIP_SAFE_CALL(hipMemGetAllocationGranularity(
                &granularity, prop, hipMemAllocationGranularityMinimum));
    *actualsize = (size + granularity - 1) & ~(granularity - 1);
}
# endif /* USE_MMAP_EXPLICITLY */

static void *
XKRT_DRIVER_ENTRYPOINT(memory_device_allocate)(
    device_driver_id_t device_driver_id,
    const size_t size,
    int area_idx
) {
    assert(area_idx == 0);

    # if USE_MMAP_EXPLICITLY
    hipMemAllocationProp prop;
    size_t actualsize;
    get_prop_and_size(device_driver_id, size, &prop, &actualsize);

    hipDeviceptr_t addr = 0;
    HIP_SAFE_CALL(hipMemAddressReserve(&addr, actualsize, 0, 0, 0));  // reserve VA space

    hipMemGenericAllocationHandle_t handle;
    HIP_SAFE_CALL(hipMemCreate(&handle, actualsize, &prop, 0));       // allocate physical memory
    HIP_SAFE_CALL(hipMemMap(addr, actualsize, 0, handle, 0));         // map it
    HIP_SAFE_CALL(hipMemRelease(handle));                             // (optional) release handle

    CUmemAccessDesc desc = {};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = device_driver_id;
    desc.flags = hipMemaccess_tFlagsProtReadWrite;
    HIP_SAFE_CALL(cuMemSetAccess(addr, actualsize, &desc, 1));

    return (void *) addr;
    # else
    hip_set_context(device_driver_id);
    hipDeviceptr_t device_ptr = (hipDeviceptr_t) NULL;
    hipMalloc(&device_ptr, size);
    return (void *) device_ptr;
    # endif
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_deallocate)(
    device_driver_id_t device_driver_id,
    void * ptr,
    const size_t size,
    int area_idx
) {
    assert(area_idx == 0);
    # if USE_MMAP_EXPLICITLY
    hipMemAllocationProp prop;
    size_t actualsize;
    get_prop_and_size(device_driver_id, size, &prop, &actualsize);
    HIP_SAFE_CALL(hipMemUnmap((hipDeviceptr_t) ptr, actualsize));
    HIP_SAFE_CALL(hipMemAddressFree((hipDeviceptr_t) ptr, actualsize));
    # else
    (void) size;
    hip_set_context(device_driver_id);
    HIP_SAFE_CALL(hipFree((hipDeviceptr_t) ptr));
    # endif
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_unified_allocate)(device_driver_id_t device_driver_id, const size_t size)
{
    (void) device_driver_id;
    hipDeviceptr_t device_ptr;
    HIP_SAFE_CALL(hipMallocManaged(&device_ptr, size, hipMemAttachGlobal));
    return (void *) device_ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_unified_deallocate)(device_driver_id_t device_driver_id, void * ptr, const size_t size)
{
    (void) device_driver_id;
    (void) size;
    HIP_SAFE_CALL(hipFree((hipDeviceptr_t) ptr));
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_info)(device_driver_id_t device_driver_id, device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX], int * nmemories)
{
    hip_set_context(device_driver_id);

    size_t free, total;
    HIP_SAFE_CALL(hipMemGetInfo(&free, &total));
    info[0].capacity = total;
    info[0].used     = total - free;
    strncpy(info[0].name, "(null)", sizeof(info[0].name));
    *nmemories = 1;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_destroy)(device_driver_id_t device_driver_id)
{
    device_hip_t * device = device_hip_get(device_driver_id);
    (void) device;
    return 0;
}

/* Called for each device of the driver once they all have been initialized */
static int
XKRT_DRIVER_ENTRYPOINT(device_commit)(
    device_driver_id_t device_driver_id,
    device_unique_id_bitfield_t * affinity
) {
    assert(affinity);

    device_hip_t * device = device_hip_get(device_driver_id);
    assert(device);
    assert(device->inherited.state == XKRT_DEVICE_STATE_INIT);

    hip_set_context(device_driver_id);

    /* all other devices have been initialized, enable peer */
    for (int other_device_driver_id = 0 ; other_device_driver_id < XKRT_DEVICES_MAX ; ++other_device_driver_id)
    {
        device_hip_t * other_device = device_hip_get(other_device_driver_id);
        assert(other_device);
        if (other_device->inherited.state < XKRT_DEVICE_STATE_INIT)
            continue ;

        /* add device with itself */
        if (device_driver_id == other_device_driver_id)
        {
            affinity[0] |= (device_unique_id_bitfield_t) (1UL << device->inherited.unique_id);
        }
        else
        {
            if (hip_use_p2p)
            {
                int access;
                HIP_SAFE_CALL(hipDeviceCanAccessPeer(&access, device->hip.device, other_device->hip.device));

                if (access)
                {
                    unsigned int flags = 0;
                    hipError_t res = hipDeviceEnablePeerAccess(other_device_driver_id, flags);
                    if ((res == hipSuccess) || (res == hipErrorPeerAccessAlreadyEnabled))
                    {
                        int rank = hip_perf_topo[device_driver_id*hip_device_count+other_device_driver_id];
                        assert(rank > 0);
                        if (hip_perf_device[device_driver_id*XKRT_DEVICES_PERF_RANK_MAX+rank] & (1UL << other_device_driver_id))
                        {
                            affinity[rank] |= (device_unique_id_bitfield_t) (1UL << other_device->inherited.unique_id);
                        }
                    }
                    else
                    {
                        LOGGER_WARN("Could not enable peer from %d to %d",
                                device->inherited.unique_id, other_device->inherited.unique_id);
                    }
                }
                else
                {
                    LOGGER_WARN("GPU peer from %d to %d is not possible",
                            device->inherited.unique_id, other_device->inherited.unique_id);
                }
            }
            else
            {
                LOGGER_WARN("GPU Peer disabled");
            }
        }
    }

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_register)(
    void * ptr,
    uint64_t size
) {
    // even though we are using `hipHostRegisterPortable` - which should
    // pin across all contextes, it seems Cuda Driver requires the current
    // thread to be bound to some context
    HIP_SAFE_CALL(hipHostRegister(ptr, size, hipHostRegisterPortable));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_unregister)(
    void * ptr,
    uint64_t size
) {
    (void) size;
    HIP_SAFE_CALL(hipHostUnregister(ptr));
    return 0;
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_host_allocate)(
    device_driver_id_t device_driver_id,
    uint64_t size
) {
    (void) device_driver_id;
    void * ptr;
    hip_set_context(device_driver_id);
    HIP_SAFE_CALL(hipHostAlloc(&ptr, size, hipHostRegisterPortable));
    // HIP_SAFE_CALL(cuHostAlloc(&ptr, size, cuHostRegisterPortable | cuHostAllocWriteCombined));
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
    HIP_SAFE_CALL(hipHostFree(mem));
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
        default:
            return 4;
    }
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_launch)(
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    assert(queue);

    hipEvent_t event = queue->hip.events.buffer[idx];
    hipStream_t handle = queue->hip.handle.high;

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
            const size_t count  = cmd->copy_1D.size;
            assert(count > 0);

            void * src = (void *) cmd->copy_1D.src_device_addr;
            void * dst = (void *) cmd->copy_1D.dst_device_addr;

            switch (cmd->type)
            {
                case (ocg::COMMAND_TYPE_COPY_H2D_1D):
                {
                    HIP_SAFE_CALL(hipMemcpyHtoDAsync((hipDeviceptr_t) dst, src, count, handle));
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_1D):
                {
                    HIP_SAFE_CALL(hipMemcpyDtoHAsync(dst, (hipDeviceptr_t) src, count, handle));
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2D_1D):
                {
                    HIP_SAFE_CALL(hipMemcpyDtoDAsync((hipDeviceptr_t) dst, (hipDeviceptr_t) src, count, handle));
                    break ;
                }

                default:
                {
                    LOGGER_FATAL("unreachable");
                    break ;
                }

            }
            HIP_SAFE_CALL(hipEventRecord(event, handle));
            return EINPROGRESS;
        }

        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
        {
            hipDeviceptr_t src_deviceptr, dst_deviceptr;
            hipMemoryType src_type, dst_type;
            void * src_host, * dst_host;

            void * src = (void *) cmd->copy_2D.src_device_view.addr;
            void * dst = (void *) cmd->copy_2D.dst_device_view.addr;

            switch (cmd->type)
            {
                case (ocg::COMMAND_TYPE_COPY_H2D_2D):
                {
                    src_type = hipMemoryTypeHost;
                    dst_type = hipMemoryTypeDevice;

                    src_deviceptr   = 0;
                    src_host        = src;

                    dst_deviceptr   = (hipDeviceptr_t) dst;
                    dst_host        = NULL;

                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_2D):
                {
                    src_type = hipMemoryTypeDevice;
                    dst_type = hipMemoryTypeHost;

                    src_deviceptr   = (hipDeviceptr_t) src;
                    src_host        = NULL;

                    dst_deviceptr   = 0;
                    dst_host        = dst;

                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2D_2D):
                {
                    src_type = hipMemoryTypeDevice;
                    dst_type = hipMemoryTypeDevice;

                    src_deviceptr   = (hipDeviceptr_t) src;
                    src_host        = NULL;

                    dst_deviceptr   = (hipDeviceptr_t) dst;
                    dst_host        = NULL;

                    break ;
                }

                default:
                {
                    LOGGER_FATAL("unreachable");
                    break ;
                }
            }

            const size_t dpitch = cmd->copy_2D.dst_device_view.ld * cmd->copy_2D.sizeof_type;
            const size_t spitch = cmd->copy_2D.src_device_view.ld * cmd->copy_2D.sizeof_type;

            const size_t width  = cmd->copy_2D.m * cmd->copy_2D.sizeof_type;
            const size_t height = cmd->copy_2D.n;
            assert(width > 0);
            assert(height > 0);

            hip_Memcpy2D cpy = {
                .srcXInBytes    = 0,
                .srcY           = 0,
                .srcMemoryType  = src_type,
                .srcHost        = src_host,
                .srcDevice      = src_deviceptr,
                .srcArray       = NULL,
                .srcPitch       = spitch,
                .dstXInBytes    = 0,
                .dstY           = 0,
                .dstMemoryType  = dst_type,
                .dstHost        = dst_host,
                .dstDevice      = dst_deviceptr,
                .dstArray       = NULL,
                .dstPitch       = dpitch,
                .WidthInBytes   = width,
                .Height         = height
            };
            HIP_SAFE_CALL(hipMemcpyParam2DAsync(&cpy, handle));
            HIP_SAFE_CALL(hipEventRecord(event, handle));
            return EINPROGRESS;
        }

        default:
            return EINVAL;
    }

    /* unreachable code */
    LOGGER_FATAL("Unreachable code");
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(
    command_queue_t * iqueue
) {
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    assert(queue);

    HIP_SAFE_CALL(hipStreamSynchronize(queue->hip.handle.high));
    HIP_SAFE_CALL(hipStreamSynchronize(queue->hip.handle.low));

    return 0;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait)(
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    assert(queue);

    assert(idx >= 0);
    assert(idx < queue->hip.events.capacity);

    hipEvent_t * event = queue->hip.events.buffer + idx;
    assert(event);

    HIP_SAFE_CALL(hipEventSynchronize(*event));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_progress)(
    command_queue_t * iqueue
) {
    assert(iqueue);
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    int r = 0;

    iqueue->progress([&] (command_t * cmd, xkrt_command_queue_list_counter_t p) {

        switch (cmd->type)
        {
            case (ocg::COMMAND_TYPE_PROG):
            case (ocg::COMMAND_TYPE_PROG_LAUNCHER):
            case (ocg::COMMAND_TYPE_COPY_H2H_1D):
            case (ocg::COMMAND_TYPE_COPY_H2D_1D):
            case (ocg::COMMAND_TYPE_COPY_D2H_1D):
            case (ocg::COMMAND_TYPE_COPY_D2D_1D):
            case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            case (ocg::COMMAND_TYPE_COPY_D2D_2D):
            {
                hipEvent_t event = queue->hip.events.buffer[p];
                hipError_t res = hipEventQuery(event);
                if (res == hipErrorNotReady)
                    r = EINPROGRESS;
                else if (res == hipSuccess)
                    iqueue->complete_command(p);
                else
                    LOGGER_FATAL("Error querying event");
                break ;
            }

            default:
                LOGGER_FATAL("Wrong command");
        }

        return true;
    });

    return r;
}

static command_queue_t *
XKRT_DRIVER_ENTRYPOINT(command_queue_create)(
    device_t * device,
    command_queue_type_t type,
    xkrt_command_queue_list_counter_t capacity
) {
    assert(device);
    hip_set_context(device->driver_id);

    uint8_t * mem = (uint8_t *) malloc(sizeof(queue_hip_t) + capacity * sizeof(hipEvent_t));
    assert(mem);

    queue_hip_t * queue = (queue_hip_t *) mem;

    /*************************/
    /* init xkrt queue */
    /*************************/
    command_queue_init(
        (command_queue_t *) queue,
        type,
        capacity,
        XKRT_DRIVER_ENTRYPOINT(command_queue_launch),
        XKRT_DRIVER_ENTRYPOINT(command_queue_progress),
        XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all),
        XKRT_DRIVER_ENTRYPOINT(command_queue_wait)
    );

    /*************************/
    /* do cu specific init */
    /*************************/

    /* events */
    queue->hip.events.buffer = (hipEvent_t *) (queue + 1);
    queue->hip.events.capacity = capacity;

    for (unsigned int i = 0 ; i < capacity ; ++i)
        HIP_SAFE_CALL(hipEventCreateWithFlags(queue->hip.events.buffer + i, hipEventDisableTiming));

    /* queues */
    int leastPriority, greatestPriority;
    HIP_SAFE_CALL(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    HIP_SAFE_CALL(hipStreamCreateWithPriority(&queue->hip.handle.high, hipStreamNonBlocking, greatestPriority));
    HIP_SAFE_CALL(hipStreamCreateWithPriority(&queue->hip.handle.low, hipStreamNonBlocking, leastPriority));

    if (type == XKRT_QUEUE_TYPE_KERN)
    {
        HIPBLAS_SAFE_CALL(hipblasCreate(&queue->hip.blas.handle));
        HIPBLAS_SAFE_CALL(hipblasSetStream(queue->hip.blas.handle, queue->hip.handle.high));
    }
    else
    {
        queue->hip.blas.handle = 0;
    }

    return (command_queue_t *) queue;
}

static void
XKRT_DRIVER_ENTRYPOINT(command_queue_delete)(
    command_queue_t * iqueue
) {
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    if (queue->hip.blas.handle)
        hipblasDestroy(queue->hip.blas.handle);
    HIP_SAFE_CALL(hipStreamDestroy(queue->hip.handle.high));
    HIP_SAFE_CALL(hipStreamDestroy(queue->hip.handle.low));
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
    device_hip_t * device = device_hip_get(device_driver_id);
    assert(device);

    snprintf(buffer, size, "%s, cu device: %i, pci: %02x:%02x, %.2f (GB)",
        device->hip.prop.name,
        device->inherited.unique_id,
        device->hip.prop.pciBusID,
        device->hip.prop.pciDeviceID,
        ((double)device->hip.prop.mem_total)/1e9
    );
}

driver_module_t
XKRT_DRIVER_ENTRYPOINT(module_load)(
    device_driver_id_t device_driver_id,
    uint8_t * bin,
    size_t binsize,
    driver_module_format_t format
) {
    (void) binsize;
    assert(format == XKRT_DRIVER_MODULE_FORMAT_NATIVE);
    hip_set_context(device_driver_id);
    driver_module_t module = NULL;
    HIP_SAFE_CALL(hipModuleLoadData((hipModule_t *) &module, bin));
    assert(module);
    return module;
}

void
XKRT_DRIVER_ENTRYPOINT(module_unload)(
    driver_module_t module
) {
    HIP_SAFE_CALL(hipModuleUnload((hipModule_t) module));
}

driver_module_fn_t
XKRT_DRIVER_ENTRYPOINT(module_get_fn)(
    driver_module_t module,
    const char * name
) {
    driver_module_fn_t fn = NULL;
    HIP_SAFE_CALL(hipModuleGetFunction((hipFunction_t *) &fn, (hipModule_t) module, name));
    assert(fn);
    return fn;
}

# if XKRT_SUPPORT_NVML
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

# endif /* XKRT_SUPPORT_NVML */
#include <hip/hip_runtime.h>

int
XKRT_DRIVER_ENTRYPOINT(transfer_h2d)(void * dst, void * src, const size_t size)
{
    HIP_SAFE_CALL(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2h)(void * dst, void * src, const size_t size)
{
    HIP_SAFE_CALL(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2d)(void * dst, void * src, const size_t size)
{
    HIP_SAFE_CALL(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_h2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    HIP_SAFE_CALL(hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, (hipStream_t)(queue->hip.handle.high)));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2h_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    HIP_SAFE_CALL(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, (hipStream_t)(queue->hip.handle.high)));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_hip_t * queue = (queue_hip_t *) iqueue;
    HIP_SAFE_CALL(hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, (hipStream_t)(queue->hip.handle.high)));
    return 0;
}

driver_t *
XKRT_DRIVER_ENTRYPOINT(create_driver)(void)
{
    driver_hip_t * driver = (driver_hip_t *) calloc(1, sizeof(driver_hip_t));
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

    REGISTER(transfer_h2d);
    REGISTER(transfer_d2h);
    REGISTER(transfer_d2d);
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
    REGISTER(memory_unified_allocate);
    REGISTER(memory_unified_deallocate);

    REGISTER(device_cpuset);

    REGISTER(command_queue_suggest);
    REGISTER(command_queue_create);
    REGISTER(command_queue_delete);

    REGISTER(module_load);
    REGISTER(module_unload);
    REGISTER(module_get_fn);

    # if XKRT_SUPPORT_NVML
    REGISTER(power_start);
    REGISTER(power_stop);
    # endif /* XKRT_SUPPORT_NVML */

    # undef REGISTER

    return (driver_t *) driver;
}

XKRT_NAMESPACE_END
