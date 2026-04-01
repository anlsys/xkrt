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

// https://docs.nvidia.com/cu/cu-driver-api/

# define XKRT_DRIVER_ENTRYPOINT(N) XKRT_DRIVER_TYPE_CU_ ## N

# include <xkrt/support.h>
# include <xkrt/driver/device.hpp>
# include <xkrt/driver/driver.h>
# include <xkrt/driver/driver-cu.h>
# include <xkrt/driver/queue.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/logger-cu.h>
# include <xkrt/logger/logger-cublas.h>
# include <xkrt/logger/logger-cusolver.h>
# include <xkrt/logger/logger-cusparse.h>
# include <xkrt/logger/logger-hwloc.h>
# include <xkrt/logger/metric.h>
# include <xkrt/sync/bits.h>
# include <xkrt/sync/mutex.h>

# include <cuda.h>
# include <cublas_v2.h>

# if XKRT_SUPPORT_NVML
#  include <nvml.h>
#  include <xkrt/logger/logger-nvml.h>
# endif /* XKRT_SUPPORT_NVML */

# include <hwloc.h>
# include <hwloc/cuda.h>
# include <hwloc/glibc-sched.h>

# include <cassert>
# include <cstdio>
# include <cstdint>
# include <cerrno>

# include <algorithm>

XKRT_NAMESPACE_BEGIN

/* number of used device for this run */
static device_cu_t DEVICES[XKRT_DEVICES_MAX];

static inline device_t *
device_get(device_driver_id_t device_driver_id)
{
    return (device_t *) (DEVICES + device_driver_id);
}

static inline device_cu_t *
device_cu_get(device_driver_id_t device_driver_id)
{
    return (device_cu_t *) device_get(device_driver_id);
}

static inline void
cu_set_context(device_driver_id_t device_driver_id)
{
    device_cu_t * device = device_cu_get(device_driver_id);
    CU_SAFE_CALL(cuCtxSetCurrent(device->cu.context));
}

static unsigned int
XKRT_DRIVER_ENTRYPOINT(get_ndevices_max)(void)
{
    int device_count = 0;
    CU_SAFE_CALL(cuDeviceGetCount(&device_count));
    return (unsigned int)device_count;
}

/* cu_perf_topo[i,j] returns the perf_rank of the communication link between
   device.
   cu_perf_device[d][i] for i=0,..,XKRT_DEVICES_PERF_RANK_MAX-1 is the mask of device
   for which the device d has link with performance i.
*/

static int                           cu_device_count   = 0;
static int *                         cu_perf_topo      = NULL;
static device_unique_id_bitfield_t * cu_perf_device    = NULL;
static bool                          cu_use_p2p        = false;

static void
get_gpu_topo(
    unsigned int ndevices,
    bool use_p2p
) {
    cu_device_count = ndevices;

    cu_perf_topo = (int *) malloc(sizeof(int) * cu_device_count * cu_device_count);
    assert(cu_perf_topo);

    int rank_used[64];
    memset(rank_used, 0, sizeof(rank_used));
    rank_used[0] = 1;

    // Enumerates Device <-> Device links and store perf_rank
    for (int i = 0; i < cu_device_count; ++i)
    {
        device_cu_t * device_i = device_cu_get(i);
        for (int j = 0; j < cu_device_count; ++j)
        {
            const int idx = i*cu_device_count+j;
            if (i == j)
            {
                // device to same device = highest perf
                cu_perf_topo[idx] = 0;
            }
            else
            {
                cu_perf_topo[idx] = INT_MAX;

                if (use_p2p)
                {
                    device_cu_t * device_j = device_cu_get(j);
                    int perf_rank = 0;
                    int access_supported = 0;

                    CU_SAFE_CALL(
                        cuDeviceGetP2PAttribute(
                            &access_supported,
                            CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED,
                            device_i->cu.device,
                            device_j->cu.device
                         )
                    );

                    if (access_supported)
                    {
                        CU_SAFE_CALL(
                            cuDeviceGetP2PAttribute(
                                &perf_rank,
                                CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK,
                                device_i->cu.device,
                                device_j->cu.device
                            )
                        );
                        cu_perf_topo[idx] = 1 + perf_rank;

                        if (1U + perf_rank >= sizeof(rank_used) / sizeof(*rank_used))
                            LOGGER_FATAL("P2P perf_rank too high");
                        rank_used[1 + perf_rank] = 1;
                        LOGGER_DEBUG("PERF FROM %d to %d is %d", i, j, perf_rank + 1);
                    }
                    else
                    {
                        LOGGER_WARN("GPU access from %d to %d is not supported", i, j);
                    }
                }
            }
        }
    }

    /* shrink perf ranks, on MI300A it starts at 4 somehow */
    for (int i = 0 ; i < cu_device_count*cu_device_count ; ++i)
    {
        if (cu_perf_topo[i] == INT_MAX)
            cu_perf_topo[i] = XKRT_DEVICES_PERF_RANK_MAX - 1;
        else
        {
            const int perf_rank = cu_perf_topo[i];
            int rank = perf_rank;
            while (rank - 1 > 0 && rank_used[rank - 1] == 0)
                --rank;

            if (rank != perf_rank)
            {
                LOGGER_DEBUG("SHRINKING PERF FROM RANK %d", perf_rank);
                for (int j = i ; j < cu_device_count*cu_device_count ; ++j)
                    if (cu_perf_topo[j] == perf_rank)
                        cu_perf_topo[j] = rank;

                rank_used[rank]      = 1;
                rank_used[perf_rank] = 0;
            }
        }

        if (cu_perf_topo[i] >= XKRT_DEVICES_PERF_RANK_MAX)
            LOGGER_FATAL("Too many perf ranks. Recompile increasing `XKRT_DEVICES_PERF_RANK_MAX` to at least %d", XKRT_DEVICES_PERF_RANK_MAX);
    }

    // get number of ranks
    size_t size = cu_device_count * XKRT_DEVICES_PERF_RANK_MAX * sizeof(device_unique_id_bitfield_t);
    cu_perf_device = (device_unique_id_bitfield_t *) malloc(size);
    assert(cu_perf_device);
    memset(cu_perf_device, 0, size);

    for (int device_cu_id = 0 ; device_cu_id < cu_device_count ; ++device_cu_id)
    {
        for (int other_device_cu_id = 0 ; other_device_cu_id < cu_device_count ; ++other_device_cu_id)
        {
            int rank = cu_perf_topo[device_cu_id*cu_device_count+other_device_cu_id];
            assert(0 <= device_cu_id * cu_device_count   + rank);
            assert(     device_cu_id * XKRT_DEVICES_PERF_RANK_MAX + rank <= cu_device_count * XKRT_DEVICES_PERF_RANK_MAX);

            cu_perf_device[device_cu_id * XKRT_DEVICES_PERF_RANK_MAX + rank] |= (1 << other_device_cu_id);
        }
    }
}

static int
XKRT_DRIVER_ENTRYPOINT(init)(
    unsigned int ndevices,
    bool use_p2p
) {
    LOGGER_INFO("Calling cuInit(0) ...");
    CUresult err = cuInit(0);
    LOGGER_INFO("Returned from cuInit(0)");
    if (err != CUDA_SUCCESS)
    {
        if (err == CUDA_ERROR_STUB_LIBRARY)
            LOGGER_WARN("Tried to load Cuda driver with a stub library...");
        return 1;
    }
    cu_use_p2p = use_p2p;

    int ndevices_max;
    err = cuDeviceGetCount(&ndevices_max);
    if (err)
        return 1;
    ndevices = MIN((int)ndevices, ndevices_max);

    // TODO : move that to device init
    assert(ndevices <= XKRT_DEVICES_MAX);
    for (unsigned int i = 0 ; i < ndevices ; ++i)
    {
        device_cu_t * device = device_cu_get(i);
        device->inherited.state = XKRT_DEVICE_STATE_DEALLOCATED;
        CU_SAFE_CALL(cuDeviceGet(&device->cu.device, i));
        // CU_SAFE_CALL(cuCtxCreate(&device->cu.context, 0, device->cu.device));
        CU_SAFE_CALL(cuDevicePrimaryCtxRetain(&device->cu.context, device->cu.device));
    }

    get_gpu_topo(ndevices, use_p2p);

    # if XKRT_SUPPORT_NVML
    NVML_SAFE_CALL(nvmlInit());

    // TODO : that shit may allow to control nvlink power use, could be interesting in the future

    // NVML_GPU_NVLINK_BW_MODE_FULL      = 0x0
    // NVML_GPU_NVLINK_BW_MODE_OFF       = 0x1
    // NVML_GPU_NVLINK_BW_MODE_MIN       = 0x2
    // NVML_GPU_NVLINK_BW_MODE_HALF      = 0x3
    // NVML_GPU_NVLINK_BW_MODE_3QUARTER  = 0x4
    // NVML_GPU_NVLINK_BW_MODE_COUNT     = 0x5
    // TODO NVML_SAFE_CALL(nvmlSystemSetNvlinkBwMode(0x3));

    # endif /* XKRT_SUPPORT_NVML */

    return 0;
}

static void
XKRT_DRIVER_ENTRYPOINT(finalize)(void)
{
    # if XKRT_SUPPORT_NVML
    NVML_SAFE_CALL(nvmlShutdown());
    # endif /* XKRT_SUPPORT_NVML */
}

static const char *
XKRT_DRIVER_ENTRYPOINT(get_name)(void)
{
    return "CUDA";
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
    HWLOC_SAFE_CALL(hwloc_cuda_get_device_cpuset(topology, device_driver_id, cpuset));

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

    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device->inherited.state == XKRT_DEVICE_STATE_DEALLOCATED);

    return (device_t *) device;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_init)(device_driver_id_t device_driver_id)
{
    cu_set_context(device_driver_id);

    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);
    assert(device->inherited.state == XKRT_DEVICE_STATE_CREATE);

    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.pciBusID,    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID,         device->cu.device));
    CU_SAFE_CALL(cuDeviceGetAttribute(&device->cu.prop.pciDeviceID, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID,      device->cu.device));

    memset(device->cu.prop.name, 0, sizeof(device->cu.prop.name));
    CU_SAFE_CALL(cuDeviceGetName(device->cu.prop.name, sizeof(device->cu.prop.name), device->cu.device));

    CU_SAFE_CALL(cuDeviceTotalMem(&device->cu.prop.mem_total, device->cu.device));
}

# define USE_MMAP_EXPLICITLY 0

# if USE_MMAP_EXPLICITLY
static inline void
get_prop_and_size(
    device_driver_id_t device_driver_id,
    const size_t size,
    CUmemAllocationProp * prop,
    size_t * actualsize
) {
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = device_driver_id;
    prop->win32HandleMetaData = NULL;
    prop->allocFlags.compressionType = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    prop->allocFlags.gpuDirectRDMACapable = 0;
    prop->allocFlags.usage = 0;
    prop->allocFlags.reserved[0] = 0;
    prop->allocFlags.reserved[1] = 0;
    prop->allocFlags.reserved[2] = 0;
    prop->allocFlags.reserved[3] = 0;

    size_t granularity;
    CU_SAFE_CALL(cuMemGetAllocationGranularity(
                &granularity, prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
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
    CUmemAllocationProp prop;
    size_t actualsize;
    get_prop_and_size(device_driver_id, size, &prop, &actualsize);

    CUdeviceptr addr = 0;
    CU_SAFE_CALL(cuMemAddressReserve(&addr, actualsize, 0, 0, 0));  // reserve VA space

    CUmemGenericAllocationHandle handle;
    CU_SAFE_CALL(cuMemCreate(&handle, actualsize, &prop, 0));       // allocate physical memory
    CU_SAFE_CALL(cuMemMap(addr, actualsize, 0, handle, 0));         // map it
    CU_SAFE_CALL(cuMemRelease(handle));                             // (optional) release handle

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device_driver_id;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CU_SAFE_CALL(cuMemSetAccess(addr, actualsize, &desc, 1));

    return (void *) addr;
    # else
    cu_set_context(device_driver_id);
    CUdeviceptr device_ptr = (CUdeviceptr) NULL;
    cuMemAlloc(&device_ptr, size);
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
    CUmemAllocationProp prop;
    size_t actualsize;
    get_prop_and_size(device_driver_id, size, &prop, &actualsize);
    CU_SAFE_CALL(cuMemUnmap((CUdeviceptr) ptr, actualsize));
    CU_SAFE_CALL(cuMemAddressFree((CUdeviceptr) ptr, actualsize));
    # else
    (void) size;
    cu_set_context(device_driver_id);
    CU_SAFE_CALL(cuMemFree((CUdeviceptr) ptr));
    # endif
}

static void *
XKRT_DRIVER_ENTRYPOINT(memory_unified_allocate)(device_driver_id_t device_driver_id, const size_t size)
{
    cu_set_context(device_driver_id);
    CUdeviceptr device_ptr;
    CU_SAFE_CALL(cuMemAllocManaged(&device_ptr, size, CU_MEM_ATTACH_GLOBAL));
    return (void *) device_ptr;
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_unified_deallocate)(device_driver_id_t device_driver_id, void * ptr, const size_t size)
{
    (void) size;
    cu_set_context(device_driver_id);
    CU_SAFE_CALL(cuMemFree((CUdeviceptr) ptr));
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_info)(device_driver_id_t device_driver_id, device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX], int * nmemories)
{
    cu_set_context(device_driver_id);

    size_t free, total;
    CU_SAFE_CALL(cuMemGetInfo(&free, &total));
    info[0].capacity = total;
    info[0].used     = total - free;
    strncpy(info[0].name, "(null)", sizeof(info[0].name));
    *nmemories = 1;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_destroy)(device_driver_id_t device_driver_id)
{
    device_cu_t * device = device_cu_get(device_driver_id);
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

    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);
    assert(device->inherited.state == XKRT_DEVICE_STATE_INIT);

    cu_set_context(device_driver_id);

    /* all other devices have been initialized, enable peer */
    for (int other_device_driver_id = 0 ; other_device_driver_id < XKRT_DEVICES_MAX ; ++other_device_driver_id)
    {
        device_cu_t * other_device = device_cu_get(other_device_driver_id);
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
            if (cu_use_p2p)
            {
                int access;
                CU_SAFE_CALL(cuDeviceCanAccessPeer(&access, device->cu.device, other_device->cu.device));

                if (access)
                {
                    CUresult res = cuCtxEnablePeerAccess(other_device->cu.context, 0);
                    if ((res == CUDA_SUCCESS) || (res == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED))
                    {
                        int rank = cu_perf_topo[device_driver_id*cu_device_count+other_device_driver_id];
                        assert(rank);
                        if (cu_perf_device[device_driver_id*XKRT_DEVICES_PERF_RANK_MAX+rank] & (1UL << other_device_driver_id))
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
    // if no context is set, set context '0'
    CUcontext ctx;
    CU_SAFE_CALL(cuCtxGetCurrent(&ctx));
    if (ctx == NULL)
        cu_set_context(0);

    // even though we are using `CU_MEMHOSTREGISTER_PORTABLE` - which should
    // pin across all contextes, it seems Cuda Driver requires the current
    // thread to be bound to some context
    CU_SAFE_CALL(cuMemHostRegister(ptr, size, CU_MEMHOSTREGISTER_PORTABLE));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(memory_host_unregister)(
    void * ptr,
    uint64_t size
) {
    (void) size;
    CU_SAFE_CALL(cuMemHostUnregister(ptr));
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
    CU_SAFE_CALL(cuMemHostAlloc(&ptr, size, CU_MEMHOSTREGISTER_PORTABLE));
    // CU_SAFE_CALL(cuHostAlloc(&ptr, size, cuHostRegisterPortable | cuHostAllocWriteCombined));
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
    CU_SAFE_CALL(cuMemFreeHost(mem));
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

/* Return a handle to the druver's internal representation of the batch */
void *
xkrt_cuda_driver_command_batch_init(
    device_driver_id_t device_driver_id,
    ocg::command_t * cmd
) {
    assert(cmd->type == ocg::COMMAND_TYPE_BATCH);
    assert(cmd->batch.cg);

    /* set context */
    cu_set_context(device_driver_id);

    /* retrieve associated device */
    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);

    /* allocate handle */
    command_batch_cu_handle_t * handle = (command_batch_cu_handle_t *) malloc(sizeof(command_batch_cu_handle_t));
    assert(handle);

    /* create a CUDA graph */
    CU_SAFE_CALL(cuGraphCreate(&handle->graph, 0));

    /* walk through the command graph from entry, in a BFS manner, so that when
     * a node is processed, all its predecessors have already been processed,
     * so we can set CUDA dependencies */
    struct pls_t { CUgraphNode cu_node; };
    using iterator_t = ocg::command_graph_t::node_iterator_t<pls_t>;
    constexpr ocg::command_graph_walk_direction_t direction          = ocg::COMMAND_GRAPH_WALK_DIRECTION_FORWARD;
    constexpr ocg::command_graph_walk_search_t    search             = ocg::COMMAND_GRAPH_WALK_SEARCH_BFS;
    constexpr bool                                include_entry_exit = false;

    std::vector<iterator_t> iterators = cmd->batch.cg->create_node_iterators<pls_t, include_entry_exit, direction, search>();

    /* Iterate once to create all nodes */
    for (iterator_t & it : iterators)
    {
        /* get command graph node */
        ocg::command_graph_node_t * node = it.node;

        /* get cugraph node */
        CUgraphNode * cu_node = &it.data.cu_node;

        /* Dependencies are pushed afterward 1 by 1 */
        const size_t ndeps = 0;
        const CUgraphNode * deps = NULL;

        /* get command */
        ocg::command_t * command = node->command;

        if (command == NULL)
        {
            assert(node->type == ocg::COMMAND_GRAPH_NODE_TYPE_CTRL);
            assert(node->type != ocg::COMMAND_GRAPH_NODE_TYPE_COMMAND);
            cudaGraphAddEmptyNode(cu_node, handle->graph, deps, ndeps);
        }
        else
        {
            assert(node->type == ocg::COMMAND_GRAPH_NODE_TYPE_COMMAND);
            switch (command->type)
            {
                case (ocg::COMMAND_TYPE_PROG):
                {
                    CUDA_KERNEL_NODE_PARAMS params;
                    memset(&params, 0, sizeof(params));
                    params.func             = (CUfunction) command->prog.launcher.variadic.fn;
                    params.gridDimX         = command->prog.grid.x;
                    params.gridDimY         = command->prog.grid.y;
                    params.gridDimZ         = command->prog.grid.z;
                    params.blockDimX        = command->prog.block.x;
                    params.blockDimY        = command->prog.block.y;
                    params.blockDimZ        = command->prog.block.z;
                    params.sharedMemBytes   = 0;
                    params.kernelParams     = NULL;
                    params.extra            = NULL;

                    /* CUDA graph kernel nodes require kernel params passed via
                     * the CUDA_KERNEL_NODE_PARAMS::extra field (same as cuLaunchKernel) */
                    void * conf[] = {
                        CU_LAUNCH_PARAM_BUFFER_POINTER,
                        command->prog.launcher.variadic.args,
                        CU_LAUNCH_PARAM_BUFFER_SIZE,
                        (void *) &command->prog.launcher.variadic.args_size,
                        CU_LAUNCH_PARAM_END
                    };
                    params.extra = conf;

                    CU_SAFE_CALL(cuGraphAddKernelNode(cu_node, handle->graph, deps, ndeps, &params));
                    break ;
                }

                case (ocg::COMMAND_TYPE_PROG_LAUNCHER):
                {
                    LOGGER_FATAL("Kernel launcher is not supported for batching with CUDA. You should provide the kernel explicitly via an `ocg::COMMAND_TYPE_PROG` to batch kernel launch.");
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_H2D_1D):
                {
                    CUDA_MEMCPY3D cpy = {0};

                    cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
                    cpy.srcHost       = (void *) command->copy_1D.src_device_addr;
                    cpy.srcPitch      = command->copy_1D.size;
                    cpy.srcHeight     = 1;

                    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                    cpy.dstDevice     = (CUdeviceptr) command->copy_1D.dst_device_addr;
                    cpy.dstPitch      = command->copy_1D.size;
                    cpy.dstHeight     = 1;

                    cpy.WidthInBytes  = command->copy_1D.size;
                    cpy.Height        = 1;
                    cpy.Depth         = 1;

                    CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));

                    break;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_1D):
                {
                    CUDA_MEMCPY3D cpy = {0};

                    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                    cpy.srcDevice     = (CUdeviceptr) command->copy_1D.dst_device_addr;
                    cpy.srcPitch      = command->copy_1D.size;
                    cpy.srcHeight     = 1;

                    cpy.dstMemoryType = CU_MEMORYTYPE_HOST;
                    cpy.dstHost       = (void *) command->copy_1D.src_device_addr;
                    cpy.dstPitch      = command->copy_1D.size;
                    cpy.dstHeight     = 1;

                    cpy.WidthInBytes  = command->copy_1D.size;
                    cpy.Height        = 1;
                    cpy.Depth         = 1;

                    CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));

                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2D_1D):
                {
                    CUDA_MEMCPY3D cpy;
                    memset(&cpy, 0, sizeof(cpy));
                    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
                    cpy.srcDevice     = (CUdeviceptr) command->copy_1D.src_device_addr;
                    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                    cpy.dstDevice     = (CUdeviceptr) command->copy_1D.dst_device_addr;
                    cpy.WidthInBytes  = command->copy_1D.size;
                    cpy.Height        = 1;
                    cpy.Depth         = 1;
                    CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_H2D_2D):
                case (ocg::COMMAND_TYPE_COPY_D2H_2D):
                case (ocg::COMMAND_TYPE_COPY_D2D_2D):
                {
                    CUmemorytype src_type, dst_type;
                    CUdeviceptr src_deviceptr = 0, dst_deviceptr = 0;
                    void * src_host = NULL, * dst_host = NULL;

                    void * src = (void *) command->copy_2D.src_addr;
                    void * dst = (void *) command->copy_2D.dst_addr;

                    switch (command->type)
                    {
                        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
                            src_type = CU_MEMORYTYPE_HOST;   src_host = src;
                            dst_type = CU_MEMORYTYPE_DEVICE; dst_deviceptr = (CUdeviceptr) dst;
                            break ;
                        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
                            src_type = CU_MEMORYTYPE_DEVICE; src_deviceptr = (CUdeviceptr) src;
                            dst_type = CU_MEMORYTYPE_HOST;   dst_host = dst;
                            break ;
                        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
                            src_type = CU_MEMORYTYPE_DEVICE; src_deviceptr = (CUdeviceptr) src;
                            dst_type = CU_MEMORYTYPE_DEVICE; dst_deviceptr = (CUdeviceptr) dst;
                            break ;
                        default:
                            LOGGER_FATAL("unreachable");
                            break ;
                    }

                    const size_t dpitch = command->copy_2D.dst_ld * command->copy_2D.sizeof_type;
                    const size_t spitch = command->copy_2D.src_ld * command->copy_2D.sizeof_type;
                    const size_t width  = command->copy_2D.m * command->copy_2D.sizeof_type;
                    const size_t height = command->copy_2D.n;

                    CUDA_MEMCPY3D cpy;
                    memset(&cpy, 0, sizeof(cpy));
                    cpy.srcMemoryType = src_type;
                    cpy.srcHost       = src_host;
                    cpy.srcDevice     = src_deviceptr;
                    cpy.srcPitch      = spitch;
                    cpy.dstMemoryType = dst_type;
                    cpy.dstHost       = dst_host;
                    cpy.dstDevice     = dst_deviceptr;
                    cpy.dstPitch      = dpitch;
                    cpy.WidthInBytes  = width;
                    cpy.Height        = height;
                    cpy.Depth         = 1;

                    CU_SAFE_CALL(cuGraphAddMemcpyNode(cu_node, handle->graph, deps, ndeps, &cpy, device->cu.context));
                    break ;
                }

                case (ocg::COMMAND_TYPE_BATCH):
                {
                    // assert(command->batch.cg == false);

                    if (command->batch.driver_handle == NULL)
                        command->batch.driver_handle = xkrt_cuda_driver_command_batch_init(device_driver_id, command);

                    if (command->batch.driver_handle == NULL)
                        LOGGER_FATAL("Failed to initialized a command batch");

                    command_batch_cu_handle_t * command_handle = (command_batch_cu_handle_t *) command->batch.driver_handle;
                    assert(command_handle);

                    CU_SAFE_CALL(cuGraphAddChildGraphNode(cu_node, handle->graph, deps, ndeps, command_handle->graph));
                    break ;
                }

                default:
                {
                    /* unsupported command type for CUDA graph batching:
                     * abort the contraction */
                    LOGGER_FATAL("Cannot batch command type %s into CUDA graph", ocg::command_type_to_str(command->type));
                    CU_SAFE_CALL(cuGraphDestroy(handle->graph));
                    return NULL;
                }
            } /* switch command->type */
        } /* if command != NULL */
    } /* for each iterator */

    assert(cmd->batch.cg);
    ocg::command_graph_node_t * entry = cmd->batch.cg->node_get_entry();

    /* iterate a second time to set dependencies */
    for (iterator_t & it : iterators)
    {
        /* get command graph node */
        ocg::command_graph_node_t * node = it.node;

        /* get cugraph node */
        CUgraphNode * cu_node = &it.data.cu_node;

        /* Add dependencies */
        for (ocg::command_graph_node_t * pred : node->predecessors)
        {
            /* must skip edges to entry, as it is not included in the cuda graph */
            if (pred == entry)
            {
                assert(pred->command == NULL);
                continue ;
            }

            CU_SAFE_CALL(cuGraphAddDependencies(
                handle->graph,
                &iterators[pred->iterator_index].data.cu_node,  // from
                cu_node,                                        // to
                1                                               // num dependencies
            ));
        }
    } /* for each iterator */

    /* instantiate the CUDA graph into an executable */
    CU_SAFE_CALL(cuGraphInstantiate(&handle->graph_exec, handle->graph, 0));

    return handle;
}

void
xkrt_cuda_driver_command_batch_deinit(
    device_driver_id_t device_driver_id,
    const ocg::command_t * cmd
) {
    command_batch_cu_handle_t * handle = (command_batch_cu_handle_t *) cmd->batch.driver_handle;

    cu_set_context(device_driver_id);

    if (handle->graph_exec)
        CU_SAFE_CALL(cuGraphExecDestroy(handle->graph_exec));

    if (handle->graph)
        CU_SAFE_CALL(cuGraphDestroy(handle->graph));

    free(handle);
}

// kernel launch
int XKRT_DRIVER_ENTRYPOINT(prog_launch)(
    command_queue_t * iqueue,
    xkrt_command_queue_list_counter_t idx,
    const driver_module_fn_t * fn,
    const unsigned int gx,
    const unsigned int gy,
    const unsigned int gz,
    const unsigned int bx,
    const unsigned int by,
    const unsigned int bz,
    const unsigned int shared_memory_bytes,
    void * args,
    const size_t args_size
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    void * conf[] = {
        CU_LAUNCH_PARAM_BUFFER_POINTER,
        args,
        CU_LAUNCH_PARAM_BUFFER_SIZE,
        (void *) &args_size,
        CU_LAUNCH_PARAM_END
    };

    CUstream handle = queue->cu.handle.high;

    CU_SAFE_CALL(
        cuLaunchKernel(
            (CUfunction) fn,
            gx, gy, gz,
            bx, by, bz,
            shared_memory_bytes,
            handle,
            nullptr,
            conf
        )
    );

    # if 1
    CUevent event = queue->cu.events.buffer[idx];
    CU_SAFE_CALL(cuEventRecord(event, handle));
    # else
    // TODO - sync on event, this is temporary to design itnerfaces
    cuStreamSynchronize(queue->cu.handle.high);
    LOGGER_ERROR("remove the sync");
    # endif

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_launch)(
    device_driver_id_t device_driver_id,
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    CUevent event = queue->cu.events.buffer[idx];
    CUstream stream = queue->cu.handle.high;

    switch (cmd->type)
    {
        case (ocg::COMMAND_TYPE_PROG):
        {
            constexpr size_t sharedmemory = 0;
            XKRT_DRIVER_ENTRYPOINT(prog_launch)(
                iqueue,
                idx,
                (const driver_module_fn_t *) cmd->prog.launcher.variadic.fn,
                cmd->prog.grid.x,
                cmd->prog.grid.y,
                cmd->prog.grid.z,
                cmd->prog.block.x,
                cmd->prog.block.y,
                cmd->prog.block.z,
                sharedmemory,
                cmd->prog.launcher.variadic.args,
                cmd->prog.launcher.variadic.args_size
            );
            return EINPROGRESS;
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
                    CU_SAFE_CALL(cuMemcpyHtoDAsync((CUdeviceptr) dst, src, count, stream));
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_1D):
                {
                    CU_SAFE_CALL(cuMemcpyDtoHAsync(dst, (CUdeviceptr) src, count, stream));
                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2D_1D):
                {
                    CU_SAFE_CALL(cuMemcpyDtoDAsync((CUdeviceptr) dst, (CUdeviceptr) src, count, stream));
                    break ;
                }

                default:
                {
                    LOGGER_FATAL("unreachable");
                    break ;
                }

            }
            CU_SAFE_CALL(cuEventRecord(event, stream));
            return EINPROGRESS;
        }

        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
        {
            CUdeviceptr src_deviceptr, dst_deviceptr;
            CUmemorytype src_type, dst_type;
            void * src_host, * dst_host;

            void * src = (void *) cmd->copy_2D.src_addr;
            void * dst = (void *) cmd->copy_2D.dst_addr;

            switch (cmd->type)
            {
                case (ocg::COMMAND_TYPE_COPY_H2D_2D):
                {
                    src_type = CU_MEMORYTYPE_HOST;
                    dst_type = CU_MEMORYTYPE_DEVICE;

                    src_deviceptr   = 0;
                    src_host        = src;

                    dst_deviceptr   = (CUdeviceptr) dst;
                    dst_host        = NULL;

                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2H_2D):
                {
                    src_type = CU_MEMORYTYPE_DEVICE;
                    dst_type = CU_MEMORYTYPE_HOST;

                    src_deviceptr   = (CUdeviceptr) src;
                    src_host        = NULL;

                    dst_deviceptr   = 0;
                    dst_host        = dst;

                    break ;
                }

                case (ocg::COMMAND_TYPE_COPY_D2D_2D):
                {
                    src_type = CU_MEMORYTYPE_DEVICE;
                    dst_type = CU_MEMORYTYPE_DEVICE;

                    src_deviceptr   = (CUdeviceptr) src;
                    src_host        = NULL;

                    dst_deviceptr   = (CUdeviceptr) dst;
                    dst_host        = NULL;

                    break ;
                }

                default:
                {
                    LOGGER_FATAL("unreachable");
                    break ;
                }
            }

            const size_t dpitch = cmd->copy_2D.dst_ld * cmd->copy_2D.sizeof_type;
            const size_t spitch = cmd->copy_2D.src_ld * cmd->copy_2D.sizeof_type;

            const size_t width  = cmd->copy_2D.m * cmd->copy_2D.sizeof_type;
            const size_t height = cmd->copy_2D.n;
            assert(width > 0);
            assert(height > 0);

            CUDA_MEMCPY2D cpy = {
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
            CU_SAFE_CALL(cuMemcpy2DAsync(&cpy, stream));
            CU_SAFE_CALL(cuEventRecord(event, stream));
            return EINPROGRESS;
        }

        case (ocg::COMMAND_TYPE_BATCH):
        {
            /* initialize cuda graph on first encounter */
            if (cmd->batch.driver_handle == NULL)
                cmd->batch.driver_handle = xkrt_cuda_driver_command_batch_init(device_driver_id, cmd);

            if (cmd->batch.driver_handle == NULL)
                LOGGER_FATAL("Failed to initialized a command batch");

            /* launch it */
            command_batch_cu_handle_t * handle = (command_batch_cu_handle_t *) cmd->batch.driver_handle;
            assert(handle->graph_exec);

            CU_SAFE_CALL(cuGraphLaunch(handle->graph_exec, stream));
            CU_SAFE_CALL(cuEventRecord(event, stream));
            return EINPROGRESS;
        }

        default:
            return EINVAL;
    }
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(
    command_queue_t * iqueue
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    CU_SAFE_CALL(cuStreamSynchronize(queue->cu.handle.high));
    CU_SAFE_CALL(cuStreamSynchronize(queue->cu.handle.low));

    return 0;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait)(
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    assert(queue);

    assert(idx < queue->cu.events.capacity);

    CUevent * event = queue->cu.events.buffer + idx;
    assert(event);

    CU_SAFE_CALL(cuEventSynchronize(*event));

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_progress)(
    command_queue_t * iqueue
) {
    assert(iqueue);

    queue_cu_t * queue = (queue_cu_t *) iqueue;
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
            case (ocg::COMMAND_TYPE_BATCH):
            {
                CUevent event = queue->cu.events.buffer[p];
                CUresult res = cuEventQuery(event);
                if (res == CUDA_ERROR_NOT_READY)
                    r = EINPROGRESS;
                else if (res == CUDA_SUCCESS)
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
    cu_set_context(device->driver_id);

    uint8_t * mem = (uint8_t *) malloc(sizeof(queue_cu_t) + capacity * sizeof(CUevent));
    assert(mem);

    queue_cu_t * queue = (queue_cu_t *) mem;

    /*************************/
    /* init xkrt queue      */
    /*************************/
    command_queue_init(
        (command_queue_t *) queue,
        type,
        capacity
    );

    /*************************/
    /* do cu specific init   */
    /*************************/

    /* events */
    queue->cu.events.buffer = (CUevent *) (queue + 1);
    queue->cu.events.capacity = capacity;

    for (unsigned int i = 0 ; i < capacity ; ++i)
        CU_SAFE_CALL(cuEventCreate(queue->cu.events.buffer + i, CU_EVENT_DISABLE_TIMING));

    /* queues */
    int leastPriority, greatestPriority;
    CU_SAFE_CALL(cuCtxGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CU_SAFE_CALL(cuStreamCreateWithPriority(&queue->cu.handle.high, CU_STREAM_NON_BLOCKING, greatestPriority));
    CU_SAFE_CALL(cuStreamCreateWithPriority(&queue->cu.handle.low, CU_STREAM_NON_BLOCKING, leastPriority));

    if (type == XKRT_QUEUE_TYPE_KERN)
    {
        CUBLAS_SAFE_CALL(cublasCreate(&queue->cu.blas.handle));
        CUBLAS_SAFE_CALL(cublasSetStream(queue->cu.blas.handle, queue->cu.handle.high));
        CUBLAS_SAFE_CALL(cublasSetMathMode(queue->cu.blas.handle, CUBLAS_TENSOR_OP_MATH));

        CUSPARSE_SAFE_CALL(cusparseCreate(&queue->cu.sparse.handle));
        CUSPARSE_SAFE_CALL(cusparseSetStream(queue->cu.sparse.handle, queue->cu.handle.high));

        CUSOLVER_SAFE_CALL(cusolverDnCreate(&queue->cu.solver.handle));
        CUSOLVER_SAFE_CALL(cusolverDnSetStream(queue->cu.solver.handle, queue->cu.handle.high));
    }
    else
    {
        queue->cu.blas.handle   = 0;
        queue->cu.sparse.handle = 0;
        queue->cu.solver.handle = 0;
    }

    return (command_queue_t *) queue;
}

static void
XKRT_DRIVER_ENTRYPOINT(command_queue_delete)(
    command_queue_t * iqueue
) {
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    if (queue->cu.blas.handle)
        cublasDestroy(queue->cu.blas.handle);
    if (queue->cu.sparse.handle)
        cusparseDestroy(queue->cu.sparse.handle);
    if (queue->cu.solver.handle)
        cusolverDnDestroy(queue->cu.solver.handle);
    CU_SAFE_CALL(cuStreamDestroy(queue->cu.handle.high));
    CU_SAFE_CALL(cuStreamDestroy(queue->cu.handle.low));
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
    device_cu_t * device = device_cu_get(device_driver_id);
    assert(device);

    snprintf(buffer, size, "%s, cu device: %i, pci: %02x:%02x, %.2f (GB)",
        device->cu.prop.name,
        device->inherited.unique_id,
        device->cu.prop.pciBusID,
        device->cu.prop.pciDeviceID,
        ((double)device->cu.prop.mem_total)/1e9
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
    cu_set_context(device_driver_id);
    driver_module_t mod = NULL;
    CU_SAFE_CALL(cuModuleLoadData((CUmodule *) &mod, bin));
    assert(mod);

    # if 0 && XKRT_SUPPORT_DEBUG
    LOGGER_DEBUG("%s", (char *) bin);

    unsigned int count = 0;
    CU_SAFE_CALL(cuModuleGetFunctionCount(&count, (CUmodule) mod));
    LOGGER_DEBUG("Module has %u functions", count);

    if (count > 0)
    {
        CUfunction * functions = (CUfunction *) malloc(sizeof(CUfunction) * count);
        assert(functions);
        CU_SAFE_CALL(cuModuleEnumerateFunctions(functions, count, (CUmodule) mod));
        for (unsigned int i = 0 ; i < count ; ++i)
        {
            CUfunction function = functions[i];

            int maxThreads = 0;
            int sharedMemBytes = 0;
            int constSizeBytes = 0;

            CU_SAFE_CALL(cuFuncGetAttribute(&maxThreads,     CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
            CU_SAFE_CALL(cuFuncGetAttribute(&sharedMemBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,     function));
            CU_SAFE_CALL(cuFuncGetAttribute(&constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,      function));

            LOGGER_DEBUG("  MaxThreadsPerBlock: %d", maxThreads);
            LOGGER_DEBUG("  SharedMemBytes: %d", sharedMemBytes);
            LOGGER_DEBUG("  ConstSizeBytes: %d", constSizeBytes);
        }
        free(functions);
    }

    # endif

    return mod;
}

void
XKRT_DRIVER_ENTRYPOINT(module_unload)(
    driver_module_t mod
) {
    CU_SAFE_CALL(cuModuleUnload((CUmodule) mod));
}

driver_module_fn_t
XKRT_DRIVER_ENTRYPOINT(module_get_fn)(
    driver_module_t mod,
    const char * name
) {
    driver_module_fn_t fn = NULL;
    CU_SAFE_CALL(cuModuleGetFunction((CUfunction *) &fn, (CUmodule) mod, name));
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

int
XKRT_DRIVER_ENTRYPOINT(transfer_h2d)(void * dst, void * src, const size_t size)
{
    CU_SAFE_CALL(cuMemcpyHtoD((CUdeviceptr) dst, src, size));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2h)(void * dst, void * src, const size_t size)
{
    CU_SAFE_CALL(cuMemcpyDtoH(dst, (CUdeviceptr) src, size));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2d)(void * dst, void * src, const size_t size)
{
    CU_SAFE_CALL(cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr) src, size));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_h2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CU_SAFE_CALL(cuMemcpyHtoDAsync((CUdeviceptr) dst, src, size, queue->cu.handle.high));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2h_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CU_SAFE_CALL(cuMemcpyDtoHAsync(dst, (CUdeviceptr) src, size, queue->cu.handle.high));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(transfer_d2d_async)(void * dst, void * src, const size_t size, command_queue_t * iqueue)
{
    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CU_SAFE_CALL(cuMemcpyDtoDAsync((CUdeviceptr)dst, (CUdeviceptr) src, size, queue->cu.handle.high));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_advise_device)(
    const xkrt_device_driver_id_t device_driver_id,
    const void * addr,
    const size_t size
) {
    // const CUmem_advise advice = CU_MEM_ADVISE_SET_ACCESSED_BY;
    const CUmem_advise advice = CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_DEVICE,
        .id   = device_driver_id
    };
    CU_SAFE_CALL(cuMemAdvise_v2((const CUdeviceptr) addr, size, advice, location));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_advise_host)(
    const void * addr,
    const size_t size
) {
    // const CUmem_advise advice = CU_MEM_ADVISE_SET_ACCESSED_BY;
    const CUmem_advise advice = CU_MEM_ADVISE_SET_PREFERRED_LOCATION;
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_HOST,
        .id   = 0 // ignored
    };
    CU_SAFE_CALL(cuMemAdvise_v2((const CUdeviceptr) addr, size, advice, location));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_prefetch_device)(
    const xkrt_device_driver_id_t device_driver_id,
    const void * addr,
    const size_t size
) {
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_DEVICE,
        .id   = device_driver_id
    };
    unsigned int flags = 0;

    // retrieve a H2D stream
    device_t * device = device_get(device_driver_id);
    assert(device);

    thread_t * thread;
    command_queue_t * iqueue;
    device->offloader_queue_next(XKRT_QUEUE_TYPE_H2D, &thread, &iqueue);
    assert(thread);
    assert(iqueue);

    queue_cu_t * queue = (queue_cu_t *) iqueue;
    CUstream stream = queue->cu.handle.high;

    CU_SAFE_CALL(cuMemPrefetchAsync_v2((const CUdeviceptr) addr, size, location, flags, stream));
    return 0;
}

int
XKRT_DRIVER_ENTRYPOINT(memory_unified_prefetch_host)(
    const void * addr,
    const size_t size
) {
    const CUmemLocation location = {
        .type = CU_MEM_LOCATION_TYPE_HOST,
        .id   = 0 // ignored
    };
    unsigned int flags = 0;
    CUstream stream = NULL;
    LOGGER_FATAL("What stream to use?");
    CU_SAFE_CALL(cuMemPrefetchAsync_v2((const CUdeviceptr) addr, size, location, flags, stream));
    return 0;
}

driver_t *
XKRT_DRIVER_ENTRYPOINT(create_driver)(void)
{
    driver_cu_t * driver = (driver_cu_t *) calloc(1, sizeof(driver_cu_t));
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

    REGISTER(prog_launch);

    REGISTER(memory_device_info);
    REGISTER(memory_device_allocate);
    REGISTER(memory_device_deallocate);
    REGISTER(memory_host_allocate);
    REGISTER(memory_host_deallocate);
    REGISTER(memory_host_register);
    REGISTER(memory_host_unregister);
    REGISTER(memory_unified_allocate);
    REGISTER(memory_unified_deallocate);
    REGISTER(memory_unified_advise_device);
    REGISTER(memory_unified_advise_host);
    REGISTER(memory_unified_prefetch_device);
    REGISTER(memory_unified_prefetch_host);

    REGISTER(device_cpuset);

    REGISTER(command_queue_create);
    REGISTER(command_queue_delete);
    REGISTER(command_queue_delete);
    REGISTER(command_queue_launch);
    REGISTER(command_queue_progress);
    REGISTER(command_queue_wait_all);
    REGISTER(command_queue_wait);

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
