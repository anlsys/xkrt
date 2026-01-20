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
# include <xkrt/memory/alignedas.h>
# include <xkrt/memory/access/blas/memory-tree.hpp>
# include <xkrt/memory/access/blas/dependency-tree.hpp>

XKRT_NAMESPACE_BEGIN

using fetch_list_t = KBLASMemoryTree<2>::fetch_list_t;
using fetch_t      = KBLASMemoryTree<2>::fetch_t;

// args for 'runtime->coherent_async'
typedef struct alignas(CACHE_LINE_SIZE) args_t
{
    runtime_t * runtime;
    std::atomic<int> counter;

    args_t(runtime_t * runtime) : runtime(runtime), counter(0) {}
    ~args_t() {}
}                                       args_t;

//////////////////////
// Memory coherency //
//////////////////////

void
runtime_t::memory_noncoherent_alloc(
    device_global_id_t device_global_id,
    void * ptr,
    size_t size
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);
    assert(thread->current_task);

    /* create an access to insert in the memory tree */
    const uintptr_t a = (const uintptr_t) ptr;
    const uintptr_t b = a + size;
    access_t access(NULL, a, b, ACCESS_MODE_V);
    BLASMemoryTree * memtree = (BLASMemoryTree *) task_get_memory_controller(this, thread->current_task, &access);
    memtree->allocate_to_device(&access, device_global_id);
}

void
runtime_t::memory_noncoherent_alloc(
    device_global_id_t device_global_id,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t sizeof_type
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);
    assert(thread->current_task);

    /* create an access to insert in the memory tree */
    access_t access(NULL, storage, ptr, ld, m, n, sizeof_type, ACCESS_MODE_V);
    BLASMemoryTree * memtree = (BLASMemoryTree *) task_get_memory_controller(this, thread->current_task, &access);
    memtree->allocate_to_device(&access, device_global_id);
}

void
runtime_t::memory_coherent_async(
    device_global_id_t device_global_id,
    void * ptr,
    size_t size
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    # define AC 1
    constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT | TASK_FLAG_DEVICE;
    constexpr size_t task_size = task_compute_size(flags, AC);
    constexpr size_t args_size = 0;
    constexpr task_format_id_t fmtid = XKRT_TASK_FORMAT_NULL;

    task_t * task = this->task_new(fmtid, flags, task_size + args_size);

    task_dep_info_t * dep = TASK_DEP_INFO(task);
    new (dep) task_dep_info_t(AC);

    task_dev_info_t * dev = TASK_DEV_INFO(task);
    new (dev) task_dev_info_t(device_global_id, UNSPECIFIED_TASK_ACCESS);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "coherent1D_async");
    # endif /* XKRT_SUPPORT_DEBUG */

    static_assert(AC <= TASK_MAX_ACCESSES);
    access_t * accesses = TASK_ACCESSES(task, flags);
    const uintptr_t a = (const uintptr_t) ptr;
    const uintptr_t b = a + size;
    new (accesses + 0) access_t(task, a, b, ACCESS_MODE_R);
    thread->resolve(accesses, AC);
    # undef AC

    this->task_commit(task);
}

void
runtime_t::memory_coherent_async(
    device_global_id_t device_global_id,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t sizeof_type
) {
    /* relying on prefetching */
    # if 1
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    # define AC 1

    constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT | TASK_FLAG_DEVICE;
    constexpr size_t task_size = task_compute_size(flags, AC);
    constexpr size_t args_size = 0;
    constexpr task_format_id_t fmtid = XKRT_TASK_FORMAT_NULL;

    task_t * task = this->task_new(fmtid, flags, task_size + args_size);

    task_dep_info_t * dep = TASK_DEP_INFO(task);
    new (dep) task_dep_info_t(AC);

    task_dev_info_t * dev = TASK_DEV_INFO(task);
    new (dev) task_dev_info_t(device_global_id, UNSPECIFIED_TASK_ACCESS);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "coherent1D_async");
    # endif /* XKRT_SUPPORT_DEBUG */

    static_assert(AC <= TASK_MAX_ACCESSES);
    access_t * accesses = TASK_ACCESSES(task, flags);
    new (accesses + 0) access_t(task, storage, ptr, ld, m, n, sizeof_type, ACCESS_MODE_R);
    thread->resolve(accesses, AC);
    # undef AC

    this->task_commit(task);

    # else

    /* parsing dependence tree (unsafe now that it may be cleared dynamically by task_new) */
    // LOGGER_IMPL("in `memory_coherent_async` - uplo and memflag parameters not supported");

    // against memory-tree, dep-tree does not know where the data will be once the predecessor task executed.
    // For instance, two continuous partites may end-up being coherent on two different devices, thus cannot be merged
    // Therefore, this impl creates 1 copy per partite (it is not even obvious that merging can improve perf.)

    thread_t * thread = thread_t::get_tls();
    assert(thread);
    assert(thread->current_task);

    /* create an access, and retrieve all dependency tree nodes that are in conflict */
    DependencyDomain * domain = task_get_dependency_domain_blas_matrix(thread->current_task, ld, sizeof_type);
    assert(domain);

    access_t access(NULL, storage, ptr, ld, m, n, sizeof_type, ACCESS_MODE_R);

    std::vector<void *> conflicts;
    ((BLASDependencyTree *) domain)->conflicting(&conflicts, &access);

    LOGGER_DEBUG("`memory_coherent_async` found %zu conflicts", conflicts.size());

    /* create one task per conflict responsible to fetch the node */
    # define AC 1
    constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT | TASK_FLAG_DEVICE;
    constexpr size_t args_size = sizeof(args_t);
    constexpr size_t task_size = task_compute_size(flags, AC);
    constexpr task_format_id_t fmtid = XKRT_TASK_FORMAT_NULL;

    /* for each node of the dep tree conflicting */
    for (void * & conflict : conflicts)
    {
        /* retrieve the node */
        BLASDependencyTree::Node * node = (BLASDependencyTree::Node *) conflict;
        access_t * write = node->last_write;
        assert(write);

        // this may no longer be true due to dependencies between segments and matrices
        // assert(access.host_view.ld          == write->host_view.ld);
        // assert(access.host_view.sizeof_type == write->host_view.sizeof_type);

        /* allocate a task with 1 access */
        task_t * task = this->task_new<false>(fmtid, flags, task_size + args_size);

        task_dev_info_t * dev = TASK_DEV_INFO(task);
        new (dev) task_dev_info_t(device_global_id, UNSPECIFIED_TASK_ACCESS);

        args_t * args = (args_t *) TASK_ARGS(task, task_size);
        new (args) args_t(this);

        task_dep_info_t * dep = TASK_DEP_INFO(task);
        new (dep) task_dep_info_t(AC);

        #if XKRT_SUPPORT_DEBUG
        strncpy(task->label, "memory_coherent_async", sizeof(task->label));
        #endif /* XKRT_SUPPORT_DEBUG */

        access_t * accesses = TASK_ACCESSES(task, flags);
        assert(accesses);

        /* as 'conflicts' are forming a partition of 'access', it must only
         * intersects with a single cube of 'access' : find which of the two */
        bool found = false;
        for (const Rect & rect : access.rects())
        {
            Rect h;
            Rect::intersection(&h, rect, node->hyperrect);

            if (!h.is_empty())
            {
                new (accesses + 0) access_t(task, MATRIX_COLMAJOR, h, access.host_view.ld, access.host_view.sizeof_type, ACCESS_MODE_R);
                __access_precedes(write, accesses + 0);
                found = true;
                break ;
            }
        }
        /* assert to check that we did find a cube from 'access' that intersects with the node */
        assert(found);

        // insert for future tasks dependencies
        domain->put<AC>(accesses);

        // commit the task
        this->task_commit(task);
    }

    # undef AC
    # endif
}

void
runtime_t::memory_coherent_sync(
    device_global_id_t device_global_id,
    void * ptr,
    size_t size
) {
    this->memory_coherent_async(device_global_id, ptr, size);
    this->task_wait();
}

void
runtime_t::memory_coherent_sync(
    device_global_id_t device_global_id,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t sizeof_type
) {
    this->memory_coherent_async(device_global_id, storage, ptr, ld, m, n, sizeof_type);
    this->task_wait();
}

XKRT_NAMESPACE_END
