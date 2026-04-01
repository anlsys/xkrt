/*
 * f2h_pipeline.cc — Pipelined, NUMA-aware File → GPU Transfer
 *
 * Three stages run concurrently and overlap via a chunk pool:
 *
 *   [free_q] ──► READER (raw io_uring syscalls, O_DIRECT, fixed-buffers)
 *            ──► [io_done_q] ──► H2D (CUDA streams + events)
 *            ──► [free_q]   (recycled)
 *
 * Compiled as C++.  All void* casts are explicit, volatile pointers are
 * cast to non-volatile before __atomic builtins, and only hwloc 2.x
 * APIs are used (hwloc_topology_get_allowed_cpuset, not the removed
 * hwloc_topology_get_online_cpuset).
 *
 * Build:  see Makefile
 * Usage:  ./f2h_pipeline <file> [cuda_device_index]
 */

#define _GNU_SOURCE
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <linux/io_uring.h>   /* kernel UAPI: structs, opcodes, offsets */

#include <cuda.h>
#include <hwloc.h>
#include <hwloc/cuda.h>

/* ═══════════════════════════ tunables ══════════════════════════════ */

#ifndef CHUNK_SIZE
#  define CHUNK_SIZE        ((size_t)(4*1024*1024))
#endif
#ifndef PIPELINE_DEPTH
#  define PIPELINE_DEPTH    1024
#endif
#ifndef N_STREAMS
#  define N_STREAMS         4
#endif

/* Must be a power-of-two ≥ 2 × PIPELINE_DEPTH */
#define URING_SQ_DEPTH      32

/* ═══════════════════════════ helpers ═══════════════════════════════ */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        CUresult _r = (call);                                               \
        if (_r != CUDA_SUCCESS) {                                           \
            const char *_s = "(unknown)";                                   \
            cuGetErrorString(_r, &_s);                                      \
            fprintf(stderr, "CUDA error @ %s:%d: %s\n",                    \
                    __FILE__, __LINE__, _s);                                \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define DIE(fmt, ...)                                                       \
    do {                                                                    \
        fprintf(stderr, "FATAL %s:%d: " fmt "\n",                          \
                __FILE__, __LINE__, ##__VA_ARGS__);                         \
        exit(EXIT_FAILURE);                                                 \
    } while (0)

#define MIN(a, b)  ((a) < (b) ? (a) : (b))

/*
 * VPTR — cast void* to T* explicitly.
 * Needed everywhere mmap/calloc/malloc return void* in C++ context.
 */
template<typename T>
static inline T *vptr(void *p) { return static_cast<T *>(p); }

/*
 * au32_load / au32_store — atomic ops on uint32_t that live behind a
 * volatile pointer.  GCC's __atomic builtins accept volatile pointers
 * in C but not always in strict C++; we cast away volatile here.
 * The memory-order semantics are unchanged — the cast is only to
 * satisfy the type system.
 */
static inline uint32_t au32_load_acq(volatile uint32_t *p)
{
    return __atomic_load_n(const_cast<uint32_t *>(p), __ATOMIC_ACQUIRE);
}
static inline uint32_t au32_load_rlx(volatile uint32_t *p)
{
    return __atomic_load_n(const_cast<uint32_t *>(p), __ATOMIC_RELAXED);
}
static inline void au32_store_rel(volatile uint32_t *p, uint32_t v)
{
    __atomic_store_n(const_cast<uint32_t *>(p), v, __ATOMIC_RELEASE);
}

/* ══════════════════ raw io_uring syscall wrappers ══════════════════ */

static int sys_io_uring_setup(unsigned entries, struct io_uring_params *p)
{
    return static_cast<int>(syscall(__NR_io_uring_setup, entries, p));
}

static int sys_io_uring_enter(int fd,
                               unsigned to_submit,
                               unsigned min_complete,
                               unsigned flags)
{
    return static_cast<int>(
        syscall(__NR_io_uring_enter,
                fd, to_submit, min_complete, flags,
                static_cast<void *>(nullptr), static_cast<size_t>(0)));
}

static int sys_io_uring_register(int fd, unsigned opcode,
                                  void *arg, unsigned nr_args)
{
    return static_cast<int>(
        syscall(__NR_io_uring_register, fd, opcode, arg, nr_args));
}

/* ═══════════════════════ uring_t — raw ring ════════════════════════
 *
 * Three mmap regions from the ring fd:
 *   IORING_OFF_SQ_RING  — SQ control ring (head/tail/mask + sq_array[])
 *   IORING_OFF_SQES     — flat SQE array we write into
 *   IORING_OFF_CQ_RING  — CQ control ring (head/tail/mask + inline cqes[])
 *
 * Memory ordering:
 *   kernel-written words (cq_tail, sq_head) : ACQUIRE load
 *   our own writes       (sq_tail, cq_head) : RELEASE store
 */
struct uring_t {
    int      fd;
    bool     iopoll;
    bool     fixed_reg;

    /* SQ */
    void                *sq_ring;
    size_t               sq_ring_sz;
    struct io_uring_sqe *sqes;
    size_t               sqes_sz;

    volatile uint32_t   *sq_head;   /* kernel advances — ACQUIRE load  */
    volatile uint32_t   *sq_tail;   /* we advance     — RELEASE store  */
    volatile uint32_t   *sq_mask;   /* sq_entries - 1, read-only       */
    uint32_t            *sq_array;  /* maps slot index to SQE index    */
    uint32_t             sq_pending;

    /* CQ */
    void                *cq_ring;
    size_t               cq_ring_sz;

    volatile uint32_t   *cq_head;   /* we advance     — RELEASE store  */
    volatile uint32_t   *cq_tail;   /* kernel advances — ACQUIRE load  */
    volatile uint32_t   *cq_mask;   /* cq_entries - 1, read-only       */
    struct io_uring_cqe *cqes;
};

static void uring_setup(uring_t *u, unsigned sq_depth)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    memset(u,  0, sizeof *u);
    u->fd = -1;

    /* Try polled first (best NVMe latency); fall back on refusal */
    p.flags = IORING_SETUP_IOPOLL;
    u->fd   = sys_io_uring_setup(sq_depth, &p);
    if (u->fd < 0) {
        memset(&p, 0, sizeof p);
        u->fd = sys_io_uring_setup(sq_depth, &p);
        if (u->fd < 0)
            DIE("io_uring_setup: %s", strerror(errno));
    }
    u->iopoll = !!(p.flags & IORING_SETUP_IOPOLL);
    printf("[uring] sq_entries=%u  cq_entries=%u  iopoll=%s\n",
           p.sq_entries, p.cq_entries,
           u->iopoll ? "yes (NVMe polled)" : "no (IRQ-driven)");

    /* ── map SQE array ─────────────────────────────────────────── */
    u->sqes_sz = static_cast<size_t>(p.sq_entries) * sizeof(struct io_uring_sqe);
    void *sqes_raw = mmap(nullptr, u->sqes_sz,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED | MAP_POPULATE,
                          u->fd, IORING_OFF_SQES);
    if (sqes_raw == MAP_FAILED)
        DIE("mmap SQES: %s", strerror(errno));
    u->sqes = vptr<struct io_uring_sqe>(sqes_raw);

    /* ── map SQ ring ────────────────────────────────────────────── */
    u->sq_ring_sz = p.sq_off.array
                  + static_cast<size_t>(p.sq_entries) * sizeof(uint32_t);
    u->sq_ring = mmap(nullptr, u->sq_ring_sz,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE,
                      u->fd, IORING_OFF_SQ_RING);
    if (u->sq_ring == MAP_FAILED)
        DIE("mmap SQ ring: %s", strerror(errno));

    /* ── map CQ ring ────────────────────────────────────────────── */
    u->cq_ring_sz = p.cq_off.cqes
                  + static_cast<size_t>(p.cq_entries) * sizeof(struct io_uring_cqe);
    u->cq_ring = mmap(nullptr, u->cq_ring_sz,
                      PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE,
                      u->fd, IORING_OFF_CQ_RING);
    if (u->cq_ring == MAP_FAILED)
        DIE("mmap CQ ring: %s", strerror(errno));

    /* ── resolve field pointers from the offset tables ─────────── */
    char *sqr   = static_cast<char *>(u->sq_ring);
    u->sq_head  = reinterpret_cast<volatile uint32_t *>(sqr + p.sq_off.head);
    u->sq_tail  = reinterpret_cast<volatile uint32_t *>(sqr + p.sq_off.tail);
    u->sq_mask  = reinterpret_cast<volatile uint32_t *>(sqr + p.sq_off.ring_mask);
    u->sq_array = reinterpret_cast<uint32_t *>         (sqr + p.sq_off.array);

    char *cqr   = static_cast<char *>(u->cq_ring);
    u->cq_head  = reinterpret_cast<volatile uint32_t *>(cqr + p.cq_off.head);
    u->cq_tail  = reinterpret_cast<volatile uint32_t *>(cqr + p.cq_off.tail);
    u->cq_mask  = reinterpret_cast<volatile uint32_t *>(cqr + p.cq_off.ring_mask);
    u->cqes     = reinterpret_cast<struct io_uring_cqe *>(cqr + p.cq_off.cqes);
}

static bool uring_register_buffers(uring_t *u, struct iovec *iov, unsigned n)
{
    return sys_io_uring_register(u->fd,
                                  IORING_REGISTER_BUFFERS,
                                  static_cast<void *>(iov), n) == 0;
}

static void uring_unregister_buffers(uring_t *u)
{
    sys_io_uring_register(u->fd, IORING_UNREGISTER_BUFFERS, nullptr, 0);
}

/*
 * uring_get_sqe — next free SQE slot; NULL if ring full.
 * Uses 1:1 mapping: sq_array[i] = i (set once here per slot).
 */
static struct io_uring_sqe *uring_get_sqe(uring_t *u)
{
    uint32_t head = au32_load_acq(u->sq_head);
    uint32_t tail = au32_load_rlx(u->sq_tail) + u->sq_pending;
    uint32_t mask = au32_load_rlx(u->sq_mask);

    if ((tail - head) > mask)
        return nullptr;   /* SQ full */

    uint32_t idx     = tail & mask;
    u->sq_array[idx] = idx;
    u->sq_pending++;
    return &u->sqes[idx];
}

/*
 * uring_submit_and_wait — publish pending SQEs (RELEASE on sq_tail)
 * and block via io_uring_enter until ≥1 CQE is ready.
 */
static void uring_submit_and_wait(uring_t *u)
{
    uint32_t new_tail = au32_load_rlx(u->sq_tail) + u->sq_pending;
    if (u->sq_pending) {
        au32_store_rel(u->sq_tail, new_tail);
        u->sq_pending = 0;
    }

    uint32_t sq_head   = au32_load_acq(u->sq_head);
    unsigned to_submit = new_tail - sq_head;

    int r = sys_io_uring_enter(u->fd, to_submit, 1,
                                IORING_ENTER_GETEVENTS);
    if (r < 0 && errno != EINTR)
        DIE("io_uring_enter: %s", strerror(errno));
}

/* uring_peek_cqe — non-blocking; NULL if CQ empty. */
static struct io_uring_cqe *uring_peek_cqe(uring_t *u)
{
    uint32_t head = au32_load_rlx(u->cq_head);
    uint32_t tail = au32_load_acq(u->cq_tail);
    if (head == tail) return nullptr;
    return &u->cqes[head & au32_load_rlx(u->cq_mask)];
}

/* uring_cqe_advance — mark current CQE consumed (RELEASE on cq_head). */
static void uring_cqe_advance(uring_t *u)
{
    uint32_t h = au32_load_rlx(u->cq_head);
    au32_store_rel(u->cq_head, h + 1);
}

static void uring_exit(uring_t *u)
{
    munmap(u->sqes,    u->sqes_sz);
    munmap(u->sq_ring, u->sq_ring_sz);
    if (u->cq_ring != u->sq_ring)
        munmap(u->cq_ring, u->cq_ring_sz);
    close(u->fd);
}

/* ═══════════════════════ concurrent queue ══════════════════════════ */

struct chunk_t;   /* forward declaration so cq_t can hold chunk_t* */

struct cq_t {
    chunk_t        **buf;
    int              cap, head, tail, count;
    bool             closed;
    pthread_mutex_t  mu;
    pthread_cond_t   not_empty;
    pthread_cond_t   not_full;
};

static void cq_init(cq_t *q, int cap)
{
    q->buf    = static_cast<chunk_t **>(calloc(static_cast<size_t>(cap),
                                               sizeof(chunk_t *)));
    if (!q->buf) DIE("calloc cq: %s", strerror(errno));
    q->cap    = cap;
    q->head = q->tail = q->count = 0;
    q->closed = false;
    pthread_mutex_init(&q->mu,        nullptr);
    pthread_cond_init (&q->not_empty, nullptr);
    pthread_cond_init (&q->not_full,  nullptr);
}

static bool cq_pop(cq_t *q, chunk_t **out)
{
    pthread_mutex_lock(&q->mu);
    while (q->count == 0 && !q->closed)
        pthread_cond_wait(&q->not_empty, &q->mu);
    bool ok = (q->count > 0);
    if (ok) {
        *out    = q->buf[q->head];
        q->head = (q->head + 1) % q->cap;
        q->count--;
        pthread_cond_signal(&q->not_full);
    }
    pthread_mutex_unlock(&q->mu);
    return ok;
}

static chunk_t *cq_try_pop(cq_t *q)
{
    pthread_mutex_lock(&q->mu);
    chunk_t *c = nullptr;
    if (q->count > 0) {
        c       = q->buf[q->head];
        q->head = (q->head + 1) % q->cap;
        q->count--;
        pthread_cond_signal(&q->not_full);
    }
    pthread_mutex_unlock(&q->mu);
    return c;
}

static void cq_push(cq_t *q, chunk_t *c)
{
    pthread_mutex_lock(&q->mu);
    while (q->count == q->cap)
        pthread_cond_wait(&q->not_full, &q->mu);
    q->buf[q->tail] = c;
    q->tail = (q->tail + 1) % q->cap;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
}

static void cq_close(cq_t *q)
{
    pthread_mutex_lock(&q->mu);
    q->closed = true;
    pthread_cond_broadcast(&q->not_empty);
    pthread_mutex_unlock(&q->mu);
}

static bool cq_is_done(cq_t *q)
{
    pthread_mutex_lock(&q->mu);
    bool done = q->closed && (q->count == 0);
    pthread_mutex_unlock(&q->mu);
    return done;
}

static void cq_destroy(cq_t *q)
{
    free(q->buf);
    pthread_mutex_destroy(&q->mu);
    pthread_cond_destroy(&q->not_empty);
    pthread_cond_destroy(&q->not_full);
}

/* ══════════════════════════ chunk_t ════════════════════════════════ */

struct chunk_t {
    int         id;
    void       *host;
    CUdeviceptr dev;
    size_t      len;
    off_t       file_off;
    CUstream    stream;
    CUevent     event;
    bool        pinned;   /* cuMemHostRegister has been called for this slot */
};

/* ════════════════════════ pipeline_t ══════════════════════════════ */

struct pipeline_t {
    const char      *path;
    int              fd;
    off_t            file_size;

    chunk_t          pool[PIPELINE_DEPTH];
    void            *host_base;
    size_t           host_base_sz;

    cq_t             free_q;
    cq_t             pin_done_q;   /* pinner  → reader  */
    cq_t             io_done_q;    /* reader  → h2d     */

    uring_t          uring;
    struct iovec     iov[PIPELINE_DEPTH];

    CUcontext        ctx;
    CUstream         streams[N_STREAMS];
    CUdeviceptr      dev_base;
    size_t           dev_sz;

    hwloc_topology_t topo;
    int              gpu_numa;
    hwloc_cpuset_t   gpu_cpuset;

    off_t            next_off;         /* reader thread only — no lock needed */
    size_t           total_chunks;     /* ceil(file_size / CHUNK_SIZE)        */
    size_t           bytes_xfer;       /* updated via __atomic_fetch_add      */

    struct timespec  ts_start, ts_end;
};

/* ═════════════════ hwloc topology + thread affinity ════════════════
 *
 * hwloc 2.x removed hwloc_topology_get_online_cpuset().
 * Use hwloc_topology_get_allowed_cpuset() instead — it returns the
 * set of CPUs the process is allowed to use, which is exactly what
 * we want when intersecting with the GPU-local CPUs.
 */
static void topo_init(pipeline_t *p, int cuda_idx)
{
    hwloc_topology_init(&p->topo);
    hwloc_topology_set_io_types_filter(p->topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    hwloc_topology_load(p->topo);

    p->gpu_cpuset = hwloc_bitmap_alloc();
    p->gpu_numa   = -1;

    hwloc_obj_t osdev =
        hwloc_cuda_get_device_osdev_by_index(p->topo, cuda_idx);
    if (!osdev) {
        fprintf(stderr, "[topo] GPU %d not in hwloc — no NUMA affinity\n",
                cuda_idx);
        hwloc_bitmap_copy(p->gpu_cpuset,
                          hwloc_topology_get_complete_cpuset(p->topo));
        return;
    }

    /* Walk up to nearest NUMA node */
    hwloc_obj_t node = osdev;
    while (node && node->type != HWLOC_OBJ_NUMANODE)
        node = node->parent;
    if (!node) {
        /* Alternative path: via PCI device */
        hwloc_obj_t pci = osdev->parent;
        while (pci && pci->type != HWLOC_OBJ_PCI_DEVICE)
            pci = pci->parent;
        if (pci) {
            hwloc_obj_t tmp = pci->parent;
            while (tmp && tmp->type != HWLOC_OBJ_NUMANODE)
                tmp = tmp->parent;
            node = tmp;
        }
    }
    if (node)
        p->gpu_numa = static_cast<int>(node->os_index);

    /* GPU-local CPU set: walk up until we find an object with a cpuset */
    hwloc_obj_t root = osdev;
    while (root && (!root->cpuset || hwloc_bitmap_iszero(root->cpuset)))
        root = root->parent;

    if (root) {
        hwloc_bitmap_copy(p->gpu_cpuset, root->cpuset);
        /* Intersect with process-allowed CPUs (hwloc 2.x API) */
        hwloc_bitmap_and(p->gpu_cpuset, p->gpu_cpuset,
                         hwloc_topology_get_allowed_cpuset(p->topo));
    } else {
        hwloc_bitmap_copy(p->gpu_cpuset,
                          hwloc_topology_get_complete_cpuset(p->topo));
    }

    char *s = nullptr;
    hwloc_bitmap_asprintf(&s, p->gpu_cpuset);
    printf("[topo] GPU %d -> NUMA node %d, CPUs: %s\n",
           cuda_idx, p->gpu_numa, s);
    free(s);
}

static void thread_pin_to_gpu(pipeline_t *p)
{
    if (hwloc_bitmap_iszero(p->gpu_cpuset)) return;
    if (hwloc_set_cpubind(p->topo, p->gpu_cpuset,
                          HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT) < 0)
        hwloc_set_cpubind(p->topo, p->gpu_cpuset, HWLOC_CPUBIND_THREAD);
}

/* ════════════════════ NUMA-local pinned host pool ═══════════════════ */

static void host_alloc(pipeline_t *p)
{
    size_t sz = static_cast<size_t>(PIPELINE_DEPTH) * CHUNK_SIZE;
    p->host_base_sz = sz;

    p->host_base = mmap(nullptr, sz,
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p->host_base == MAP_FAILED)
        DIE("mmap host pool: %s", strerror(errno));

    if (p->gpu_numa >= 0) {
        hwloc_nodeset_t ns = hwloc_bitmap_alloc();
        hwloc_bitmap_set(ns, static_cast<unsigned>(p->gpu_numa));
        if (hwloc_set_area_membind(p->topo, p->host_base, sz, ns,
                                   HWLOC_MEMBIND_BIND,
                                   HWLOC_MEMBIND_BYNODESET |
                                   HWLOC_MEMBIND_MIGRATE) < 0)
            perror("[mem] hwloc membind (non-fatal)");
        hwloc_bitmap_free(ns);
    }

    /* Force physical page allocation on the target NUMA node now.
     * cuMemHostRegister will be called per-chunk inside the pipeline. */
    memset(p->host_base, 0, sz);

    printf("[mem]  %zu MiB mmap'd on NUMA node %d (pinning deferred to pipeline)\n",
           sz >> 20, p->gpu_numa);
}

/* ════════════════════════ CUDA init ═════════════════════════════════ */

static void cuda_init(pipeline_t *p, int dev_idx)
{
    CUDA_CHECK(cuInit(0));

    int ndev = 0;
    CUDA_CHECK(cuDeviceGetCount(&ndev));
    if (dev_idx >= ndev)
        DIE("device %d requested, only %d present", dev_idx, ndev);

    CUdevice dev;
    CUDA_CHECK(cuDeviceGet(&dev, dev_idx));

    char   name[256];
    size_t vmem;
    cuDeviceGetName(name, sizeof name, dev);
    cuDeviceTotalMem(&vmem, dev);
    printf("[cuda] device %d: %s (%.1f GiB)\n",
           dev_idx, name, vmem / 1073741824.0);

    CUDA_CHECK(cuCtxCreate(&p->ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));

    for (int i = 0; i < N_STREAMS; i++)
        CUDA_CHECK(cuStreamCreate(&p->streams[i], CU_STREAM_NON_BLOCKING));

    p->dev_sz = static_cast<size_t>(p->file_size);
    CUDA_CHECK(cuMemAlloc(&p->dev_base, p->dev_sz));
    printf("[cuda] %.2f GiB allocated on device\n",
           p->dev_sz / 1073741824.0);
}

/* ════════════════════════ io_uring init ═════════════════════════════ */

static void uring_init(pipeline_t *p)
{
    uring_setup(&p->uring, URING_SQ_DEPTH);

    for (int i = 0; i < PIPELINE_DEPTH; i++) {
        p->iov[i].iov_base = static_cast<char *>(p->host_base)
                           + static_cast<size_t>(i) * CHUNK_SIZE;
        p->iov[i].iov_len  = CHUNK_SIZE;
    }
    p->uring.fixed_reg =
        uring_register_buffers(&p->uring, p->iov, PIPELINE_DEPTH);
    printf("[uring] fixed buffers: %s\n",
           p->uring.fixed_reg ? "registered (IORING_OP_READ_FIXED)"
                              : "unavailable, using IORING_OP_READ");
}

/* ════════════════════════ chunk pool ═══════════════════════════════ */

static void chunks_init(pipeline_t *p)
{
    p->total_chunks = (static_cast<size_t>(p->file_size) + CHUNK_SIZE - 1)
                    / CHUNK_SIZE;

    cq_init(&p->free_q,     PIPELINE_DEPTH + 2);
    cq_init(&p->pin_done_q, PIPELINE_DEPTH + 2);
    cq_init(&p->io_done_q,  PIPELINE_DEPTH + 2);

    for (int i = 0; i < PIPELINE_DEPTH; i++) {
        chunk_t *c  = &p->pool[i];
        c->id       = i;
        c->host     = static_cast<char *>(p->host_base)
                    + static_cast<size_t>(i) * CHUNK_SIZE;
        c->dev      = 0;
        c->len      = 0;
        c->file_off = 0;
        c->stream   = nullptr;
        c->pinned   = false;
        CUDA_CHECK(cuEventCreate(&c->event, CU_EVENT_DISABLE_TIMING));
        cq_push(&p->free_q, c);
    }
}

/* ════════════════════════ PINNER THREAD ════════════════════════════
 *
 * Stage 1 of the pipeline: free_q → cuMemHostRegister → pin_done_q.
 *
 * Each chunk slot is registered exactly once (checked via chunk->pinned).
 * On subsequent passes through this stage the chunk is already pinned
 * and the thread simply forwards it — making this stage a near-zero-cost
 * passthrough after the first PIPELINE_DEPTH chunks.
 *
 * The thread exits after forwarding exactly total_chunks items, then
 * closes pin_done_q so the reader thread terminates cleanly.
 */
static void *pinner_thread(void *arg)
{
    pipeline_t *p        = static_cast<pipeline_t *>(arg);
    size_t      forwarded = 0;

    CUDA_CHECK(cuCtxSetCurrent(p->ctx));
    thread_pin_to_gpu(p);
    printf("[pinner] started (tid=%ld)\n", static_cast<long>(gettid()));

    while (forwarded < p->total_chunks) {
        chunk_t *ck = nullptr;
        if (!cq_pop(&p->free_q, &ck))
            break;   /* free_q closed unexpectedly */

        if (!ck->pinned) {
            /* First time this slot is used: pin it.
             * Each chunk is registered individually so we can unregister
             * per-slot cleanly in pipeline_destroy.                       */
            CUDA_CHECK(cuMemHostRegister(ck->host, CHUNK_SIZE,
                                         CU_MEMHOSTREGISTER_PORTABLE |
                                         CU_MEMHOSTREGISTER_DEVICEMAP));
            ck->pinned = true;
        }

        cq_push(&p->pin_done_q, ck);
        forwarded++;
    }

    cq_close(&p->pin_done_q);
    printf("[pinner] finished (%zu chunks pinned)\n", forwarded);
    return nullptr;
}

/* ════════════════════════ READER THREAD ════════════════════════════
 *
 * Stage 2: pin_done_q → io_uring read → io_done_q.
 * Chunks arriving here are already CUDA-pinned by the pinner thread.
 */
static void *reader_thread(void *arg)
{
    pipeline_t *p          = static_cast<pipeline_t *>(arg);
    uring_t    *u          = &p->uring;
    int         in_flight   = 0;
    bool        all_sub     = false;

    thread_pin_to_gpu(p);
    printf("[reader] started (tid=%ld)\n", static_cast<long>(gettid()));

    for (;;) {
        /* ── prepare and enqueue new reads ─────────────────────── */
        while (!all_sub) {
            chunk_t *ck = cq_try_pop(&p->pin_done_q);
            if (!ck) {
                /* Queue temporarily empty — check if pinner is fully done */
                if (cq_is_done(&p->pin_done_q)) all_sub = true;
                break;
            }

            if (p->next_off >= p->file_size) {
                /* Shouldn't happen: pinner forwards exactly total_chunks */
                cq_push(&p->free_q, ck);
                all_sub = true;
                break;
            }

            ck->len      = MIN(CHUNK_SIZE,
                               static_cast<size_t>(p->file_size - p->next_off));
            ck->file_off = p->next_off;
            ck->dev      = p->dev_base + static_cast<CUdeviceptr>(p->next_off);
            p->next_off += static_cast<off_t>(ck->len);

            struct io_uring_sqe *sqe = uring_get_sqe(u);
            if (!sqe)
                DIE("SQ ring full (URING_SQ_DEPTH=%d too small)", URING_SQ_DEPTH);

            /* Zero first — kernel rejects SQEs with dirty reserved fields */
            memset(sqe, 0, sizeof *sqe);

            if (u->fixed_reg) {
                sqe->opcode    = IORING_OP_READ_FIXED;
                sqe->buf_index = static_cast<__u16>(ck->id);
            } else {
                sqe->opcode    = IORING_OP_READ;
            }
            sqe->fd        = static_cast<__s32>(p->fd);
            sqe->off       = static_cast<__u64>(static_cast<uint64_t>(ck->file_off));
            sqe->addr      = static_cast<__u64>(reinterpret_cast<uintptr_t>(ck->host));
            sqe->len       = static_cast<__u32>(ck->len);
            sqe->user_data = static_cast<__u64>(reinterpret_cast<uintptr_t>(ck));
            in_flight++;
        }

        if (in_flight == 0) {
            if (all_sub) break;
            sched_yield();
            continue;
        }

        /* ── publish SQEs and wait for ≥1 CQE ──────────────────── */
        uring_submit_and_wait(u);

        /* ── drain all ready CQEs ───────────────────────────────── */
        struct io_uring_cqe *cqe;
        while ((cqe = uring_peek_cqe(u)) != nullptr) {
            chunk_t *ck = reinterpret_cast<chunk_t *>(
                              static_cast<uintptr_t>(cqe->user_data));
            if (cqe->res < 0)
                DIE("io_uring read error at offset %ld: %s",
                    static_cast<long>(ck->file_off),
                    strerror(-static_cast<int>(cqe->res)));
            if (static_cast<size_t>(cqe->res) < ck->len)
                ck->len = static_cast<size_t>(cqe->res);

            uring_cqe_advance(u);
            cq_push(&p->io_done_q, ck);
            in_flight--;
        }
    }

    cq_close(&p->io_done_q);
    printf("[reader] finished\n");
    return nullptr;
}

/* ════════════════════════ H2D THREAD ═══════════════════════════════ */

static void *h2d_thread(void *arg)
{
    pipeline_t *p          = static_cast<pipeline_t *>(arg);
    chunk_t    *inflight[PIPELINE_DEPTH];
    int         n_inflight  = 0;
    int         stream_rr   = 0;
    bool        io_closed   = false;

    CUDA_CHECK(cuCtxSetCurrent(p->ctx));
    thread_pin_to_gpu(p);
    printf("[h2d]   started (tid=%ld)\n", static_cast<long>(gettid()));

    for (;;) {
        chunk_t *ck = nullptr;
        if (!io_closed) {
            if (n_inflight == 0) {
                if (!cq_pop(&p->io_done_q, &ck))
                    io_closed = true;
            } else {
                ck = cq_try_pop(&p->io_done_q);
                if (!ck && cq_is_done(&p->io_done_q))
                    io_closed = true;
            }
        }

        if (ck) {
            CUstream s = p->streams[stream_rr++ % N_STREAMS];
            ck->stream = s;
            CUDA_CHECK(cuMemcpyHtoDAsync(ck->dev, ck->host, ck->len, s));
            CUDA_CHECK(cuEventRecord(ck->event, s));
            inflight[n_inflight++] = ck;
        }

        /* Non-blocking sweep of in-flight H2D events */
        int still = 0;
        for (int i = 0; i < n_inflight; i++) {
            CUresult r = cuEventQuery(inflight[i]->event);
            if (r == CUDA_SUCCESS) {
                __atomic_fetch_add(&p->bytes_xfer,
                                   inflight[i]->len, __ATOMIC_RELAXED);
                cq_push(&p->free_q, inflight[i]);
            } else if (r == CUDA_ERROR_NOT_READY) {
                inflight[still++] = inflight[i];
            } else {
                CUDA_CHECK(r);
            }
        }
        n_inflight = still;

        if (io_closed && n_inflight == 0) break;
        if (!ck && n_inflight > 0) sched_yield();
    }

    printf("[h2d]   finished (%.3f GiB transferred)\n",
           static_cast<double>(p->bytes_xfer) / 1073741824.0);
    return nullptr;
}

/* ════════════════════════════ run ══════════════════════════════════ */

static void pipeline_run(pipeline_t *p)
{
    clock_gettime(CLOCK_MONOTONIC, &p->ts_start);

    pthread_t tid_pin, tid_r, tid_h;
    if (pthread_create(&tid_pin, nullptr, pinner_thread, p) != 0)
        DIE("pthread_create pinner: %s", strerror(errno));
    if (pthread_create(&tid_r,   nullptr, reader_thread, p) != 0)
        DIE("pthread_create reader: %s", strerror(errno));
    if (pthread_create(&tid_h,   nullptr, h2d_thread,    p) != 0)
        DIE("pthread_create h2d: %s", strerror(errno));

    pthread_join(tid_pin, nullptr);
    pthread_join(tid_r,   nullptr);
    pthread_join(tid_h,   nullptr);

    clock_gettime(CLOCK_MONOTONIC, &p->ts_end);
}

/* ════════════════════════ teardown ════════════════════════════════ */

static void pipeline_destroy(pipeline_t *p)
{
    for (int i = 0; i < PIPELINE_DEPTH; i++)
        cuEventDestroy(p->pool[i].event);
    for (int i = 0; i < N_STREAMS; i++)
        cuStreamDestroy(p->streams[i]);
    cuMemFree(p->dev_base);
    cuCtxDestroy(p->ctx);

    if (p->uring.fixed_reg)
        uring_unregister_buffers(&p->uring);
    uring_exit(&p->uring);

    /* Unregister each chunk slot individually — they were registered
     * one by one by the pinner thread, not as a single region.        */
    for (int i = 0; i < PIPELINE_DEPTH; i++) {
        if (p->pool[i].pinned)
            cuMemHostUnregister(p->pool[i].host);
    }
    munmap(p->host_base, p->host_base_sz);
    close(p->fd);

    hwloc_bitmap_free(p->gpu_cpuset);
    hwloc_topology_destroy(p->topo);
    cq_destroy(&p->free_q);
    cq_destroy(&p->pin_done_q);
    cq_destroy(&p->io_done_q);
}

/* ════════════════════════ statistics ══════════════════════════════ */

static void print_stats(const pipeline_t *p)
{
    double dt  = static_cast<double>(p->ts_end.tv_sec  - p->ts_start.tv_sec)
               + static_cast<double>(p->ts_end.tv_nsec - p->ts_start.tv_nsec) * 1e-9;
    double gib = static_cast<double>(p->bytes_xfer) / 1073741824.0;
    printf("\n+------------------------------------------+\n");
    printf("|  Pipeline stats                          |\n");
    printf("+------------------------------------------+\n");
    printf("|  Transferred   : %8.3f GiB            |\n", gib);
    printf("|  Wall time     : %8.3f s              |\n", dt);
    printf("|  Throughput    : %8.2f GiB/s          |\n", gib / dt);
    printf("+------------------------------------------+\n");
    printf("|  Chunk size    : %6lu MiB              |\n", CHUNK_SIZE / 1024 /1024);
    printf("|  Pipeline depth: %6d chunks           |\n", PIPELINE_DEPTH);
    printf("|  CUDA streams  : %6d                  |\n", N_STREAMS);
    printf("+------------------------------------------+\n");
}

/* ═════════════════════════════ main ════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <file> [cuda_device_index]\n\n"
                "Compile-time knobs (override via -D):\n"
                "  CHUNK_SIZE      default 4 MiB\n"
                "  PIPELINE_DEPTH  default 12 chunks\n"
                "  N_STREAMS       default 4 CUDA streams\n",
                argv[0]);
        return 1;
    }

    const char *path    = argv[1];
    int         dev_idx = (argc >= 3) ? atoi(argv[2]) : 0;

    pipeline_t p;
    memset(&p, 0, sizeof p);
    p.path   = path;
    p.fd     = -1;
    p.gpu_numa = -1;

    p.fd = open(path, O_RDONLY | O_DIRECT);
    if (p.fd < 0) {
        fprintf(stderr, "[warn] O_DIRECT unavailable (%s), falling back\n",
                strerror(errno));
        p.fd = open(path, O_RDONLY);
        if (p.fd < 0)
            DIE("open(%s): %s", path, strerror(errno));
    }

    struct stat st;
    if (fstat(p.fd, &st) < 0) DIE("fstat: %s", strerror(errno));
    p.file_size = st.st_size;
    if (p.file_size == 0) DIE("file is empty");

    printf("[main]  %s (%.3f GiB), CUDA device %d\n\n",
           path, static_cast<double>(p.file_size) / 1073741824.0, dev_idx);

    topo_init   (&p, dev_idx);   /* hwloc: find GPU NUMA node + CPU set         */
    cuda_init   (&p, dev_idx);   /* cuInit + context: must precede any cu*()    */
    host_alloc  (&p);            /* mmap + membind (no pinning yet)             */
    uring_init  (&p);            /* io_uring_setup + register fixed buffers     */
    chunks_init (&p);            /* per-chunk events + seed free_q              */

    printf("\n[main]  starting pipeline...\n\n");
    pipeline_run    (&p);
    print_stats     (&p);
    pipeline_destroy(&p);
    return 0;
}
