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
# include <xkrt/memory/access/blas/dependency-tree.hpp>

XKRT_NAMESPACE_BEGIN

# include <math.h>

////////////////
// DISTRIBUTE //
////////////////

static inline void
distribute1D_submit(
    runtime_t * runtime,
    uintptr_t x1, uintptr_t x2,
    device_unique_id_t device_unique_id
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    constexpr task_flag_bitfield_t  flags       = TASK_FLAG_ACCESSES | TASK_FLAG_DEVICE;
    constexpr void                * args        = NULL;
    constexpr size_t                args_size   = 0;
    constexpr task_format_id_t      fmtid       = XKRT_TASK_FORMAT_NULL;
    constexpr task_access_counter_t AC          = 1;

    task_t * task = runtime->task_new(fmtid, flags, args, args_size, AC);

    task_acs_info_t * acs = TASK_ACS_INFO(task);
    new (acs) task_acs_info_t(AC);

    task_dev_info_t * dev = TASK_DEV_INFO(task);
    new (dev) task_dev_info_t(device_unique_id, XKRT_UNSPECIFIED_TASK_ACCESS);

    access_t * accesses = TASK_ACCESSES(task);
    new (accesses + 0) access_t(task, x1, x2, ACCESS_MODE_R);

    runtime->task_accesses_resolve(accesses, AC);
    # undef AC

    #if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "distribute1d_async");
    #endif /* XKRT_SUPPORT_DEBUG */

    runtime->task_commit(task);
}

static inline void
distribute2D_submit(
    runtime_t * runtime,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t mb, size_t nb,
    size_t sizeof_type,
    size_t hx, size_t hy,
    size_t tm, size_t tn,
    device_unique_id_t device_unique_id
) {
    thread_t * thread = thread_t::get_tls();
    assert(thread);

    constexpr task_flag_bitfield_t  flags       = TASK_FLAG_ACCESSES | TASK_FLAG_DEVICE;
    constexpr void                * args        = NULL;
    constexpr size_t                args_size   = 0;
    constexpr task_access_counter_t AC          = 1;
    constexpr task_format_id_t      fmtid       = XKRT_TASK_FORMAT_NULL;

    task_t * task = runtime->task_new(fmtid, flags, args, args_size, AC);

    task_acs_info_t * acs = TASK_ACS_INFO(task);
    new (acs) task_acs_info_t(AC);

    task_dev_info_t * dev = TASK_DEV_INFO(task);
    new (dev) task_dev_info_t(device_unique_id, XKRT_UNSPECIFIED_TASK_ACCESS);

    access_t * accesses = TASK_ACCESSES(task);
    {
        const ssize_t  x = tm * mb;
        const ssize_t  y = tn * nb;
        const ssize_t x0 = MAX(x-(ssize_t)hx, 0);
        const ssize_t y0 = MAX(y-(ssize_t)hy, 0);
        const ssize_t x1 = MIN(x+mb+hx, m);
        const ssize_t y1 = MIN(y+nb+hy, n);
        const  size_t sx = x1 - x0;
        const  size_t sy = y1 - y0;
        new (accesses + 0) access_t(task, storage, ptr, ld, x0, y0, sx, sy, sizeof_type, ACCESS_MODE_R);
    }
    runtime->task_accesses_resolve(accesses, AC);
    # undef AC

    #if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "distribute2D_async");
    #endif /* XKRT_SUPPORT_DEBUG */

    runtime->task_commit(task);
}

static void
distribute2D_async(
    runtime_t * runtime,
    distribution_type_t type,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t mb, size_t nb,
    size_t sizeof_type,
    size_t hx, size_t hy
) {
    const int ngpus = runtime->get_ndevices() - 1;
    assert(ngpus);

    distribution_t d;
    distribution2D_init(&d, type, ngpus, m, n, mb, nb);

    for (size_t tm = 0; tm < d.mt; ++tm)
        for (size_t tn = 0; tn < d.nt; ++tn)
            distribute2D_submit(runtime, storage, ptr, ld,
                    m, n, mb, nb, sizeof_type, hx, hy, tm, tn,
                    distribution2D_get(&d, tm, tn));
}

void
runtime_t::distribute_async(
    distribution_type_t type,
    matrix_storage_t storage,
    void * ptr, size_t ld,
    size_t m, size_t n,
    size_t mb, size_t nb,
    size_t sizeof_type,
    size_t hx, size_t hy
) {
    distribute2D_async(this, type, storage, ptr, ld, m, n, mb, nb, sizeof_type, hx, hy);
}

static void
distribute1D_async(
    runtime_t * runtime,
    distribution_type_t type,
    void * ptr,
    size_t size,
    size_t chunk_size,
    size_t h
) {
    assert(type == XKRT_DISTRIBUTION_TYPE_CYCLIC1D);
    assert(h < chunk_size);

    const int ngpus = runtime->get_ndevices() - 1;
    assert(ngpus);

    distribution_t d;
    distribution1D_init(&d, type, ngpus, size, chunk_size);

    const uintptr_t p = (const uintptr_t) ptr;
    for (size_t t = 0; t < d.t; ++t)
    {
        const size_t i = (t == 0)       ?    0 : ((t+0) * chunk_size - h);
        const size_t j = (t == d.t - 1) ? size : ((t+1) * chunk_size + h);
        distribute1D_submit(runtime, p + i, p + j, distribution1D_get(&d, t));
    }
}

void
runtime_t::distribute_async(
    distribution_type_t type,
    void * ptr,
    size_t size,
    size_t bs,
    size_t h
) {
    distribute1D_async(this, type, ptr, size, bs, h);
}

static void
distribute1D_array_async(
    runtime_t * runtime,
    distribution_type_t type,
    void ** ptr,
    size_t chunk_size,
    unsigned int n
) {
    assert(type == XKRT_DISTRIBUTION_TYPE_CYCLIC1D);

    const int ngpus = runtime->get_ndevices() - 1;
    assert(ngpus);

    distribution_t d;
    distribution1D_init(&d, type, ngpus, n * chunk_size, chunk_size);

    for (size_t t = 0; t < d.t; ++t)
    {
        const uintptr_t x1 = (const uintptr_t) ptr[t];
        const uintptr_t x2 = (const uintptr_t) x1 + chunk_size;
        distribute1D_submit(runtime, x1, x2, distribution1D_get(&d, t));
    }
}

void
runtime_t::distribute_async(
    distribution_type_t type,
    void ** ptr,
    size_t chunk_size,
    unsigned int n
) {
    distribute1D_array_async(this, type, ptr, chunk_size, n);
}

XKRT_NAMESPACE_END
