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

#ifndef __XKRT_MEMORY_ALLOCATOR_BUDDY_HPP__
# define __XKRT_MEMORY_ALLOCATOR_BUDDY_HPP__

# include <xkrt/memory/allocator/allocator.hpp>
# include <xkrt/consts.h>

XKRT_NAMESPACE_BEGIN

/**
 *  Buddy allocator.
 *
 *  Power-of-two block splitting/coalescing strategy.
 *  The backing pool is rounded up to the nearest power of two.
 *  Blocks are split in half when a smaller size is needed, and
 *  coalesced with their buddy when both halves are free.
 *
 *  Manages one buddy system per device memory area.
 *  Lazily allocates backing device memory on first allocate_on() per area.
 *
 *  Minimum block size: BUDDY_MIN_BLOCK_SIZE (256 bytes).
 *  Maximum order: log2(pool_size / BUDDY_MIN_BLOCK_SIZE).
 *  Maximum supported orders: BUDDY_MAX_ORDERS (40), supporting pools
 *  up to 256 * 2^39 ~ 140TB.
 */
class buddy_allocator_t : public allocator_t
{
    public:
        /* minimum block size (must be power of 2, >= 8 for alignment) */
        static constexpr size_t BUDDY_MIN_BLOCK_SIZE = 256;

        /* maximum number of orders (supports pools up to ~140TB) */
        static constexpr int BUDDY_MAX_ORDERS = 40;

    private:

        /**
         *  Per-area buddy system state.
         */
        struct buddy_area_t
        {
            /* synchronization lock */
            xkrt_mutex_t lock;

            /* base device pointer for this area */
            uintptr_t base_ptr;

            /* total allocated size from the driver (may be < pool_size) */
            size_t alloc_size;

            /* pool size rounded up to power of 2 */
            size_t pool_size;

            /* number of orders: max_order + 1 */
            int norders;

            /**
             *  Free list heads per order.
             *  _free_lists[k] is a singly-linked list (via freelink) of
             *  free area_chunk_t blocks of size (BUDDY_MIN_BLOCK_SIZE << k).
             *  Order 0 = smallest blocks, order (norders-1) = entire pool.
             */
            area_chunk_t * free_lists[BUDDY_MAX_ORDERS];

            /**
             *  Split bitmap per order.
             *  For order k, bit i indicates that exactly one of the two
             *  buddies at index i has been allocated (XOR semantics).
             *  This is used to determine whether coalescing is possible.
             *
             *  Number of buddy pairs at order k = pool_size / (BUDDY_MIN_BLOCK_SIZE << (k+1))
             *  (no bitmap needed for the top order since there is no parent).
             *
             *  Dynamically allocated array of uint8_t bitmaps per order.
             */
            uint8_t * split_bitmaps[BUDDY_MAX_ORDERS];
        };

        /* per-area buddy state */
        buddy_area_t _buddy_areas[XKRT_DEVICE_MEMORIES_MAX];

        /**
         *  Lazily allocate backing device memory for the given area.
         *  Thread-safe (uses area lock for synchronization).
         */
        void _lazy_init(int area_idx);

        /**
         *  Initialize the buddy system for the given area.
         *  Internal helper called from _lazy_init.
         */
        void _init_area(int area_idx, uintptr_t base_ptr, size_t alloc_size);

        /**
         *  Free all split bitmaps for an area.
         */
        void _free_bitmaps(int area_idx);

        /**
         *  Compute the byte size for a memory_size_t relative to a given capacity.
         */
        static size_t _compute_size(const memory_size_t & ms, size_t capacity);

        /**
         *  Round up to the next power of two.
         */
        static size_t _next_power_of_two(size_t v);

        /**
         *  Compute the order needed for a given size.
         *  Returns the smallest k such that (BUDDY_MIN_BLOCK_SIZE << k) >= size.
         */
        static int _size_to_order(size_t size);

        /**
         *  Toggle the split bitmap bit for a block at a given order.
         *  Returns the new value of the bit (0 or 1).
         */
        static int _bitmap_toggle(uint8_t * bitmap, size_t index);

        /**
         *  Remove a chunk from a free list.
         *  Returns true if the chunk was found and removed.
         */
        static bool _free_list_remove(area_chunk_t ** head, area_chunk_t * chunk);

    public:
        buddy_allocator_t(
            memory_size_t                    memory_size_initial,
            memory_size_t                    memory_size_resize,
            f_memory_device_allocate_t       f_alloc,
            f_memory_device_deallocate_t     f_dealloc,
            device_driver_id_t               device_driver_id,
            int                              nmemories,
            const size_t                   * capacities
        );

        ~buddy_allocator_t();

        area_chunk_t * allocate_on(const size_t size, int area_idx)      override;
        area_chunk_t * allocate(const size_t size)                       override;
        void           deallocate_on(area_chunk_t * chunk, int area_idx) override;
        void           deallocate(area_chunk_t * chunk)                  override;
        void           reset(void)                                       override;
        void           reset_on(int area_idx)                            override;
};

XKRT_NAMESPACE_END

#endif /* __XKRT_MEMORY_ALLOCATOR_BUDDY_HPP__ */
