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

#ifndef __XKRT_MEMORY_ALLOCATOR_FREELIST_HPP__
# define __XKRT_MEMORY_ALLOCATOR_FREELIST_HPP__

# include <xkrt/memory/allocator/allocator.hpp>
# include <xkrt/consts.h>

XKRT_NAMESPACE_BEGIN

/**
 *  Freelist allocator.
 *
 *  Best-fit allocation strategy with chunk splitting and coalescing.
 *  Manages one area_t per device memory area.
 *
 *  Lazily allocates backing device memory on first allocate_on() per area.
 *  The first backing allocation uses memory_size_initial, subsequent ones
 *  use memory_size_resize.
 */
class freelist_allocator_t : public allocator_t
{
    private:
        /**
         *  A backing region tracks a single device memory allocation
         *  obtained via _f_alloc, so it can be released via _f_dealloc.
         */
        struct backing_region_t
        {
            uintptr_t ptr;
            size_t    size;
        };

        /**
         *  Per-area state: the area_t (lock + free_chunk_list)
         *  and list of backing regions.
         *
         *  An area is considered initialized when _nbacking[area_idx] > 0.
         */
        area_t _areas[XKRT_DEVICE_MEMORIES_MAX];

        /* backing regions per area — dynamically grown */
        backing_region_t * _backing[XKRT_DEVICE_MEMORIES_MAX];
        int                _nbacking[XKRT_DEVICE_MEMORIES_MAX];
        int                _backing_capacity[XKRT_DEVICE_MEMORIES_MAX];

        /**
         *  Lazily allocate backing device memory for the given area.
         *  Called internally on first allocate_on() for that area.
         *  Thread-safe (uses area lock for synchronization).
         */
        void _lazy_init(int area_idx);

        /**
         *  Add a new backing region to the given area.
         *  Allocates device memory via _f_alloc, creates a free chunk
         *  covering the entire region, and appends it to the free list.
         *  The first call uses memory_size_initial, subsequent calls use
         *  memory_size_resize.  If the driver allocation fails, the size
         *  is halved repeatedly until it succeeds or reaches zero.
         *  Must be called with the area lock held.
         *  Returns true on success, false if allocation ultimately failed.
         */
        bool _add_backing_region(int area_idx);

        /**
         *  Free all chunks (allocated and free) in the ordered prev/next
         *  list for the given area.
         *  Must be called with the area lock held.
         */
        void _free_area_chunks(int area_idx);

        /**
         *  Compute the byte size for a memory_size_t relative to a given capacity.
         */
        static size_t _compute_size(const memory_size_t & ms, size_t capacity);

    public:
        freelist_allocator_t(
            memory_size_t                    memory_size_initial,
            memory_size_t                    memory_size_resize,
            f_memory_device_allocate_t       f_alloc,
            f_memory_device_deallocate_t     f_dealloc,
            device_driver_id_t               device_driver_id,
            int                              nmemories,
            const size_t                   * capacities
        );

        ~freelist_allocator_t();

        area_chunk_t * allocate_on(const size_t size, int area_idx)      override;
        area_chunk_t * allocate(const size_t size)                       override;
        void           deallocate_on(area_chunk_t * chunk, int area_idx) override;
        void           deallocate(area_chunk_t * chunk)                  override;
        void           reset(void)                                       override;
        void           reset_on(int area_idx)                            override;
};

XKRT_NAMESPACE_END

#endif /* __XKRT_MEMORY_ALLOCATOR_FREELIST_HPP__ */
