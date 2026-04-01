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
    device_unique_id_t device_unique_id,
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
    memtree->allocate_to_device(&access, device_unique_id);
}

void
runtime_t::memory_noncoherent_alloc(
    device_unique_id_t device_unique_id,
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
    memtree->allocate_to_device(&access, device_unique_id);
}

constexpr task_flag_bitfield_t  flags       = TASK_FLAG_ACCESSES | TASK_FLAG_DEVICE;
constexpr void                * args        = NULL;
constexpr size_t                args_size   = 0;
constexpr task_format_id_t      fmtid       = XKRT_TASK_FORMAT_NULL;
constexpr task_access_counter_t AC          = 1;
static_assert(AC <= XKRT_TASK_MAX_ACCESSES);

static inline void
xkrt_memory_coherent_async(
    runtime_t * runtime,
    device_unique_id_t device_unique_id,
    const uintptr_t a,
    const uintptr_t b
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task_t * task           = runtime->task_new(fmtid, flags, args, args_size, AC);
    task_acs_info_t * acs   = TASK_ACS_INFO(task);
    task_dev_info_t * dev   = TASK_DEV_INFO(task);
    access_t * access       = TASK_ACCESSES(task);

    new (acs)       task_acs_info_t(AC);
    new (dev)       task_dev_info_t(device_unique_id, XKRT_UNSPECIFIED_TASK_ACCESS);
    new (access)    access_t(task, a, b, ACCESS_MODE_R);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "coherent1D_async");
    # endif /* XKRT_SUPPORT_DEBUG */

    runtime->task_accesses_resolve(access, AC);
    runtime->task_commit(task);
}

void
runtime_t::memory_coherent_async(
    device_unique_id_t device_unique_id,
    void * ptr,
    size_t size
) {
    const uintptr_t a = (uintptr_t) ptr;
    const uintptr_t b = (uintptr_t) (a + size);
    return xkrt_memory_coherent_async(this, device_unique_id, a, b);
}

void
runtime_t::memory_coherent_async(
    device_unique_id_t device_unique_id,
    void * ptr,
    size_t size,
    int n
) {
    this->foreach((uintptr_t) ptr, size, n, [&] (const int i, const uintptr_t a, const uintptr_t b)
    {
        xkrt_memory_coherent_async(this, device_unique_id, a, b);
    });
}

void
runtime_t::memory_coherent_async(
    device_unique_id_t device_unique_id,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t sizeof_type
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    task_t          * task    = this->task_new(fmtid, flags, args, args_size, AC);
    task_acs_info_t * acs     = TASK_ACS_INFO(task);
    task_dev_info_t * dev     = TASK_DEV_INFO(task);
    access_t        * access  = TASK_ACCESSES(task);

    new (acs)       task_acs_info_t(AC);
    new (dev)       task_dev_info_t(device_unique_id, XKRT_UNSPECIFIED_TASK_ACCESS);
    new (access)    access_t(task, storage, ptr, ld, m, n, sizeof_type, ACCESS_MODE_R);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "coherent2D_async");
    # endif /* XKRT_SUPPORT_DEBUG */

    this->task_accesses_resolve(access, AC);
    this->task_commit(task);
}

void
runtime_t::memory_coherent_sync(
    device_unique_id_t device_unique_id,
    void * ptr,
    size_t size
) {
    this->memory_coherent_async(device_unique_id, ptr, size);
    this->task_wait();
}

void
runtime_t::memory_coherent_sync(
    device_unique_id_t device_unique_id,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t sizeof_type
) {
    this->memory_coherent_async(device_unique_id, storage, ptr, ld, m, n, sizeof_type);
    this->task_wait();
}

XKRT_NAMESPACE_END
