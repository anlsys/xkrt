/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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

#define XKRT_DRIVER_ENTRYPOINT(N) XKRT_DRIVER_TYPE_HOST_##N

#include <xkrt/runtime.h>
#include <xkrt/conf/conf.h>
#include <xkrt/driver/device.hpp>
#include <xkrt/driver/driver.h>
#include <xkrt/driver/driver-host.h>
#include <xkrt/driver/queue.h>
#include <xkrt/sync/bits.h>
#include <xkrt/sync/mutex.h>

#include <hwloc.h>
#include <hwloc/glibc-sched.h>
#include <sys/sysinfo.h>

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cerrno>
#include <functional>

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <linux/fs.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdatomic.h>

#include <linux/io_uring.h>

XKRT_NAMESPACE_BEGIN

/* ─── Cached hwloc topology (initialized once) ─── */
static hwloc_topology_t g_topology = nullptr;
static char g_cpu_name[256] = {0};

static void
ensure_topology(void)
{
    if (__builtin_expect(g_topology != nullptr, 1))
        return;

    hwloc_topology_init(&g_topology);
    hwloc_topology_load(g_topology);

    hwloc_obj_t obj = hwloc_get_obj_by_type(g_topology, HWLOC_OBJ_PACKAGE, 0);
    if (obj && obj->name)
        snprintf(g_cpu_name, sizeof(g_cpu_name), "%s", obj->name);
    else
        snprintf(g_cpu_name, sizeof(g_cpu_name), "Unknown CPU");
}

/* ─── io_uring barrier macros ─── */
#define io_uring_smp_store_release(p, v)    \
    (reinterpret_cast<std::atomic<std::remove_reference_t<decltype(*(p))>>*>(p) \
        ->store((v), std::memory_order_release))

#define io_uring_smp_load_acquire(p)        \
    (reinterpret_cast<std::atomic<std::remove_reference_t<decltype(*(p))>>*>(p) \
        ->load(std::memory_order_acquire))

/* ─── Driver entrypoints ─── */

static int
XKRT_DRIVER_ENTRYPOINT(init)(
    unsigned int ndevices,
    bool use_p2p
) {
    (void)ndevices;
    (void)use_p2p;
    ensure_topology();
    return 0;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_info)(
    device_driver_id_t device_driver_id,
    char *buffer,
    size_t size
) {
    (void)device_driver_id;
    ensure_topology();
    snprintf(buffer, size, "%s", g_cpu_name);
}

static void
XKRT_DRIVER_ENTRYPOINT(finalize)(void)
{
    if (g_topology)
    {
        hwloc_topology_destroy(g_topology);
        g_topology = nullptr;
    }
}

static const char *
XKRT_DRIVER_ENTRYPOINT(get_name)(void)
{
    return "HOST";
}

static unsigned int
XKRT_DRIVER_ENTRYPOINT(get_ndevices_max)(void)
{
    return 1;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_cpuset)(
    hwloc_topology_t topology,
    cpu_set_t *schedset,
    device_driver_id_t device_driver_id
) {
    (void)topology;
    assert(device_driver_id == 0);
    pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), schedset);
    return 0;
}

static device_t *
XKRT_DRIVER_ENTRYPOINT(device_create)(
    driver_t *driver,
    device_driver_id_t device_driver_id
) {
    (void)driver;
    assert(device_driver_id == 0);
    static device_t device;
    return &device;
}

static void
XKRT_DRIVER_ENTRYPOINT(device_init)(device_driver_id_t device_driver_id)
{
    (void)device_driver_id;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_destroy)(device_driver_id_t device_driver_id)
{
    (void)device_driver_id;
    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(device_commit)(
    device_driver_id_t device_driver_id,
    device_unique_id_bitfield_t *affinity
) {
    (void)device_driver_id;
    (void)affinity;
    return 0;
}

//////////////////////////////////////
// QUEUE - io_uring based async I/O //
//////////////////////////////////////

static inline void
XKRT_DRIVER_ENTRYPOINT(io_uring_init)(queue_host_t * queue)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    queue->io_uring.fd = (int) syscall(__NR_io_uring_setup, XKRT_IO_URING_DEPTH, &p);
    if (queue->io_uring.fd < 0)
        LOGGER_FATAL("io_uring_setup failed: %s", strerror(errno));

    /* Compute ring buffer sizes */
    size_t sring_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cring_sz = p.cq_off.cqes  + p.cq_entries * sizeof(struct io_uring_cqe);

    const bool single_mmap = (p.features & IORING_FEAT_SINGLE_MMAP);
    if (single_mmap)
    {
        if (cring_sz > sring_sz)
            sring_sz = cring_sz;
        cring_sz = sring_sz;
    }

    /* Map submission ring */
    queue->io_uring.sq_ptr = mmap(0, sring_sz,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, queue->io_uring.fd, IORING_OFF_SQ_RING);
    if (queue->io_uring.sq_ptr == MAP_FAILED)
        LOGGER_FATAL("Failed to mmap io_uring SQ ring");

    /* Map completion ring */
    if (single_mmap)
    {
        queue->io_uring.cq_ptr = queue->io_uring.sq_ptr;
    } else {
        queue->io_uring.cq_ptr = mmap(0, cring_sz,
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, queue->io_uring.fd, IORING_OFF_CQ_RING);
        if (queue->io_uring.cq_ptr == MAP_FAILED)
            LOGGER_FATAL("Failed to mmap io_uring CQ ring");
    }

    /* Cache ring pointers — computed once, used on every submit/complete */
    char *sq = (char *)queue->io_uring.sq_ptr;
    queue->io_uring.sq_tail  = (unsigned *)(sq + p.sq_off.tail);
    queue->io_uring.sq_mask  = (unsigned *)(sq + p.sq_off.ring_mask);
    queue->io_uring.sq_array = (unsigned *)(sq + p.sq_off.array);
    queue->io_uring.sq_flags = (unsigned *)(sq + p.sq_off.flags);

    /* Map SQE array */
    queue->io_uring.sqes = (struct io_uring_sqe *)mmap(0,
        p.sq_entries * sizeof(struct io_uring_sqe),
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, queue->io_uring.fd, IORING_OFF_SQES);
    if (queue->io_uring.sqes == MAP_FAILED)
        LOGGER_FATAL("Failed to mmap io_uring SQEs");

    char * cq = (char *)queue->io_uring.cq_ptr;
    queue->io_uring.cq_head = (unsigned *)(cq + p.cq_off.head);
    queue->io_uring.cq_tail = (unsigned *)(cq + p.cq_off.tail);
    queue->io_uring.cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
    queue->io_uring.cqes    = (struct io_uring_cqe *)(cq + p.cq_off.cqes);

    /* Pre-fill the sq_array with identity mapping (index == slot).
     * This never changes for our usage pattern, so do it once. */
    for (unsigned i = 0; i < p.sq_entries; ++i)
        queue->io_uring.sq_array[i] = i;

    queue->io_uring.pending_submits = 0;
}

/*
 * Submit any SQEs that have been queued but not yet submitted to the kernel.
 */
static inline void
io_uring_flush_submits(queue_host_t * queue)
{
    unsigned pending = queue->io_uring.pending_submits;
    if (pending == 0)
        return;

    unsigned enter_flags = 0;
    int r = (int)syscall(__NR_io_uring_enter, queue->io_uring.fd, pending, 0, enter_flags, NULL, 0);
    if (r < 0)
        LOGGER_FATAL("io_uring_enter (submit) failed: %s", strerror(errno));

    queue->io_uring.pending_submits = 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_launch)(
    device_driver_id_t device_driver_id,
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    (void)device_driver_id;

    assert(
        cmd->type == ocg::COMMAND_TYPE_FD_READ  ||
        cmd->type == ocg::COMMAND_TYPE_FD_WRITE ||
        cmd->type == ocg::COMMAND_TYPE_BATCH
    );

    queue_host_t * queue = (queue_host_t *)iqueue;

    // enqueue iouring events
    if (cmd->type == ocg::COMMAND_TYPE_FD_READ || cmd->type == ocg::COMMAND_TYPE_FD_WRITE)
    {
        if (__builtin_expect(queue->io_uring.sq_ptr == NULL, 0))
            XKRT_DRIVER_ENTRYPOINT(io_uring_init)(queue);

        /* Add SQE to the tail of the ring */
        unsigned tail  = * queue->io_uring.sq_tail;
        unsigned index = tail & * queue->io_uring.sq_mask;
        struct io_uring_sqe *sqe = &queue->io_uring.sqes[index];

        /* Zero-fill and populate — clearing avoids stale flags from prior use */
        memset(sqe, 0, sizeof(*sqe));
        sqe->opcode    = (cmd->type == ocg::COMMAND_TYPE_FD_READ)
                             ? IORING_OP_READ
                             : IORING_OP_WRITE;
        sqe->fd        = cmd->file.fd;
        sqe->addr      = (unsigned long)cmd->file.buffer;
        sqe->len       = cmd->file.size;
        sqe->off       = cmd->file.offset;
        sqe->user_data = (__u64)idx;

        /* sq_array is pre-filled with identity mapping in init */

        io_uring_smp_store_release(queue->io_uring.sq_tail, tail + 1);
        ++queue->io_uring.pending_submits;

        /*
         * Batch heuristic: flush when we've accumulated enough SQEs or when
         * the ring is getting full.  This amortises the syscall cost across
         * multiple commands.
         */
        constexpr unsigned BATCH_THRESHOLD = 16;
        if (queue->io_uring.pending_submits >= BATCH_THRESHOLD)
            io_uring_flush_submits(queue);

        return EINPROGRESS;
    }
    // batch commands = emit all sub-cg commands
    else
    {
        assert(cmd->type == ocg::COMMAND_TYPE_BATCH);
        assert(cmd->batch.cg);
        LOGGER_FATAL("TODO");
    }

    return 0;
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_suggest)(
    device_driver_id_t device_driver_id,
    command_queue_type_t qtype
) {
    (void)device_driver_id;
    switch (qtype)
    {
        case (XKRT_QUEUE_TYPE_FD_READ):
        case (XKRT_QUEUE_TYPE_FD_WRITE):
            return 1;

        // KERN is used for batches
        case (XKRT_QUEUE_TYPE_KERN):
            return 1;

        default:
            return 0;
    }
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(
    command_queue_t * iqueue
) {
    assert(iqueue);
    queue_host_t * queue = (queue_host_t *)iqueue;

    assert(queue->super.type == XKRT_QUEUE_TYPE_FD_READ ||
           queue->super.type == XKRT_QUEUE_TYPE_FD_WRITE);

    /* Ensure everything has been submitted before waiting */
    io_uring_flush_submits(queue);

    int min_completion = queue->super.pending.size();
    if (min_completion)
    {
        LOGGER_DEBUG("Waiting for %d i/o commands to complete", min_completion);
        int r = (int)syscall(__NR_io_uring_enter,
                             queue->io_uring.fd, 0, min_completion,
                             IORING_ENTER_GETEVENTS, NULL, 0);
        if (r < 0)
            LOGGER_FATAL("io_uring_enter (wait) failed: %s", strerror(errno));
    }
    return 0;
}

static inline int
XKRT_DRIVER_ENTRYPOINT(command_queue_wait)(
    command_queue_t * iqueue,
    command_t * cmd,
    xkrt_command_queue_list_counter_t idx
) {
    (void)cmd;
    (void)idx;
    assert(cmd);
    // TODO: wait for specific CQE via user_data matching
    return XKRT_DRIVER_ENTRYPOINT(command_queue_wait_all)(iqueue);
}

static int
XKRT_DRIVER_ENTRYPOINT(command_queue_progress)(
    command_queue_t * iqueue
) {
    assert(iqueue);
    queue_host_t * queue = (queue_host_t *)iqueue;

    /* Flush any pending submissions so completions can arrive */
    io_uring_flush_submits(queue);

    /*
     * Drain the CQ ring.  Read tail once, then process all available
     * entries — avoids re-reading the tail on every iteration.
     */
    unsigned head = io_uring_smp_load_acquire(queue->io_uring.cq_head);
    unsigned tail = io_uring_smp_load_acquire(queue->io_uring.cq_tail);
    const unsigned mask = * queue->io_uring.cq_mask;

    if (head == tail)
        return 0;

    unsigned completed = 0;

    while (head != tail)
    {
        struct io_uring_cqe * cqe = &queue->io_uring.cqes[head & mask];

        if (__builtin_expect(cqe->res < 0, 0))
            LOGGER_FATAL("io_uring CQE error: %s", strerror(abs(cqe->res)));

        const xkrt_command_queue_list_counter_t p =
            (const xkrt_command_queue_list_counter_t)cqe->user_data;
        assert(cqe->res == (int)iqueue->pending.cmd[p].file.size);

        ++head;
        ++completed;

        /* Complete the command (callback may enqueue more work) */
        iqueue->complete_command(p);
    }

    /* Single store-release to advance head past all processed CQEs */
    io_uring_smp_store_release(queue->io_uring.cq_head, head);

    return 0;
}

static command_queue_t *
XKRT_DRIVER_ENTRYPOINT(command_queue_create)(
    device_t * idevice,
    command_queue_type_t type,
    xkrt_command_queue_list_counter_t capacity
) {
    (void)idevice;

    assert(
        type == XKRT_QUEUE_TYPE_FD_READ  ||
        type == XKRT_QUEUE_TYPE_FD_WRITE ||
        type == XKRT_QUEUE_TYPE_KERN
    );

    queue_host_t * queue = (queue_host_t *) calloc(1, sizeof(queue_host_t));
    if (!queue)
        LOGGER_FATAL("Failed to allocate queue_host_t");

    command_queue_init((command_queue_t *)queue, type, capacity);

    return (command_queue_t *)queue;
}

static void
XKRT_DRIVER_ENTRYPOINT(command_queue_delete)(
    command_queue_t * iqueue
) {
    /* TODO: munmap the io_uring ring buffers and close the fd */
    queue_host_t * queue = (queue_host_t *)iqueue;
    if (queue->io_uring.fd > 0)
        close(queue->io_uring.fd);
    free(iqueue);
}

////////////
// MEMORY //
////////////

static void *
XKRT_DRIVER_ENTRYPOINT(memory_device_allocate)(
    device_driver_id_t device_driver_id,
    const size_t size,
    int area_idx
) {
    assert(device_driver_id == 0 && area_idx == 0);
    return malloc(size);
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_deallocate)(
    device_driver_id_t device_driver_id,
    void *ptr,
    const size_t size,
    int area_idx
) {
    assert(device_driver_id == 0 && area_idx == 0);
    (void)size;
    free(ptr);
}

static void
XKRT_DRIVER_ENTRYPOINT(memory_device_info)(
    device_driver_id_t device_driver_id,
    device_memory_info_t info[XKRT_DEVICE_MEMORIES_MAX],
    int * nmemories
) {
    (void)device_driver_id;
    assert(device_driver_id == 0);

    struct sysinfo sinfo;
    if (sysinfo(&sinfo) == 0)
    {
        strncpy(info[0].name, "RAM", sizeof(info[0].name));
        info[0].used     = sinfo.totalram - sinfo.freeram;
        info[0].capacity = sinfo.totalram;
        *nmemories = 1;
    } else {
        *nmemories = 0;
    }
}

/////////////////////////
// Driver registration //
/////////////////////////

driver_t *
XKRT_DRIVER_ENTRYPOINT(create_driver)(void)
{
    driver_t * driver = (driver_t *) calloc(1, sizeof(driver_t));
    assert(driver);

    # define REGISTER(func) driver->f_##func = XKRT_DRIVER_ENTRYPOINT(func)

    REGISTER(command_queue_create);
    REGISTER(command_queue_delete);
    REGISTER(command_queue_launch);
    REGISTER(command_queue_progress);
    REGISTER(command_queue_suggest);
    REGISTER(command_queue_wait);
    REGISTER(command_queue_wait_all);

    REGISTER(device_commit);
    REGISTER(device_cpuset);
    REGISTER(device_create);
    REGISTER(device_destroy);
    REGISTER(device_info);
    REGISTER(device_init);

    REGISTER(finalize);

    REGISTER(get_name);
    REGISTER(get_ndevices_max);

    REGISTER(init);

    REGISTER(memory_device_allocate);
    REGISTER(memory_device_deallocate);
    REGISTER(memory_device_info);

    # undef REGISTER

    return driver;
}

XKRT_NAMESPACE_END
