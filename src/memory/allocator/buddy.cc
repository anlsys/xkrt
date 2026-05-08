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

# include <xkrt/memory/allocator/buddy.hpp>
# include <xkrt/logger/logger.h>

# include <cassert>
# include <cstring>
# include <cstdlib>

XKRT_NAMESPACE_USE;

///////////////////////////
// STATIC HELPERS        //
///////////////////////////

/* static */ size_t
buddy_allocator_t::_compute_size(const memory_size_t & ms, size_t capacity)
{
    if (ms.unit == XKRT_MEMORY_SIZE_UNIT_RELATIVE)
        return (size_t) ((double)capacity * (double)ms.amount / (double)MEMORY_SIZE_TYPE_MAX);
    else
        return ms.amount;
}

/* static */ size_t
buddy_allocator_t::_next_power_of_two(size_t v)
{
    if (v == 0)
        return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
}

/* static */ int
buddy_allocator_t::_size_to_order(size_t size)
{
    /* round up size to at least BUDDY_MIN_BLOCK_SIZE */
    if (size < BUDDY_MIN_BLOCK_SIZE)
        size = BUDDY_MIN_BLOCK_SIZE;

    /* round up to next power of 2 */
    size = _next_power_of_two(size);

    /* compute order: how many times we shift BUDDY_MIN_BLOCK_SIZE left */
    int order = 0;
    size_t block_size = BUDDY_MIN_BLOCK_SIZE;
    while (block_size < size)
    {
        block_size <<= 1;
        ++order;
    }
    return order;
}

/* static */ int
buddy_allocator_t::_bitmap_toggle(uint8_t * bitmap, size_t index)
{
    size_t byte_idx = index / 8;
    size_t bit_idx  = index % 8;
    bitmap[byte_idx] ^= (1 << bit_idx);
    return (bitmap[byte_idx] >> bit_idx) & 1;
}

/* static */ bool
buddy_allocator_t::_free_list_remove(area_chunk_t ** head, area_chunk_t * chunk)
{
    area_chunk_t * prev = NULL;
    area_chunk_t * curr = *head;
    while (curr)
    {
        if (curr == chunk)
        {
            if (prev)
                prev->freelink = curr->freelink;
            else
                *head = curr->freelink;
            curr->freelink = NULL;
            return true;
        }
        prev = curr;
        curr = curr->freelink;
    }
    return false;
}

///////////////////////////
// CONSTRUCTOR/DESTRUCTOR//
///////////////////////////

buddy_allocator_t::buddy_allocator_t(
    memory_size_t                    memory_size_initial,
    memory_size_t                    memory_size_resize,
    f_memory_device_allocate_t       f_alloc,
    f_memory_device_deallocate_t     f_dealloc,
    device_driver_id_t               device_driver_id,
    int                              nmemories,
    const size_t                   * capacities
)
    : allocator_t(
        memory_size_initial,
        memory_size_resize,
        f_alloc,
        f_dealloc,
        device_driver_id,
        nmemories,
        capacities
    )
{
    memset(this->_buddy_areas, 0, sizeof(this->_buddy_areas));
    for (int i = 0 ; i < XKRT_DEVICE_MEMORIES_MAX ; ++i)
        XKRT_MUTEX_INIT(this->_buddy_areas[i].lock);
}

///////////////////////////
// AREA INITIALIZATION   //
///////////////////////////

void
buddy_allocator_t::_init_area(int area_idx, uintptr_t base_ptr, size_t alloc_size)
{
    buddy_area_t * ba = &(this->_buddy_areas[area_idx]);

    ba->base_ptr   = base_ptr;
    ba->alloc_size = alloc_size;

    /* round up to power of 2 for the buddy system */
    ba->pool_size = _next_power_of_two(alloc_size);

    /*
     * If the pool_size rounded up exceeds alloc_size, shrink to the largest
     * power of 2 that fits within alloc_size.  We cannot use memory we
     * didn't actually allocate.
     */
    if (ba->pool_size > alloc_size)
        ba->pool_size = ba->pool_size >> 1;

    /* if pool_size is too small, just use alloc_size directly
     * (will be at least BUDDY_MIN_BLOCK_SIZE after the guard below) */
    if (ba->pool_size < BUDDY_MIN_BLOCK_SIZE)
        ba->pool_size = BUDDY_MIN_BLOCK_SIZE;

    assert(ba->pool_size <= alloc_size);
    assert((ba->pool_size & (ba->pool_size - 1)) == 0); /* must be power of 2 */

    /* compute number of orders */
    ba->norders = 0;
    {
        size_t sz = ba->pool_size;
        while (sz > BUDDY_MIN_BLOCK_SIZE)
        {
            sz >>= 1;
            ba->norders++;
        }
        ba->norders++; /* include order 0 */
    }
    assert(ba->norders > 0);
    assert(ba->norders <= BUDDY_MAX_ORDERS);

    /* initialize free lists */
    for (int k = 0 ; k < BUDDY_MAX_ORDERS ; ++k)
        ba->free_lists[k] = NULL;

    /* allocate split bitmaps
     * At order k, the number of buddy pairs = pool_size / (BUDDY_MIN_BLOCK_SIZE << (k+1))
     * No bitmap needed for the highest order (no parent to coalesce into).
     */
    for (int k = 0 ; k < BUDDY_MAX_ORDERS ; ++k)
        ba->split_bitmaps[k] = NULL;

    for (int k = 0 ; k < ba->norders - 1 ; ++k)
    {
        size_t block_size_at_k = BUDDY_MIN_BLOCK_SIZE << k;
        size_t npairs = ba->pool_size / (block_size_at_k * 2);
        if (npairs == 0)
            npairs = 1;
        size_t nbytes = (npairs + 7) / 8;
        ba->split_bitmaps[k] = (uint8_t *) malloc(nbytes);
        assert(ba->split_bitmaps[k]);
        memset(ba->split_bitmaps[k], 0, nbytes);
    }

    /* add the entire pool as a single free block at the highest order */
    int top_order = ba->norders - 1;
    area_chunk_t * top_chunk = (area_chunk_t *) malloc(sizeof(area_chunk_t));
    assert(top_chunk);
    top_chunk->ptr         = base_ptr;
    top_chunk->size        = ba->pool_size;
    top_chunk->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
    top_chunk->prev        = NULL;
    top_chunk->next        = NULL;
    top_chunk->freelink    = NULL;
    top_chunk->use_counter = 0;
    top_chunk->area_idx    = area_idx;

    ba->free_lists[top_order] = top_chunk;
}

void
buddy_allocator_t::_free_bitmaps(int area_idx)
{
    buddy_area_t * ba = &(this->_buddy_areas[area_idx]);
    for (int k = 0 ; k < BUDDY_MAX_ORDERS ; ++k)
    {
        if (ba->split_bitmaps[k])
        {
            free(ba->split_bitmaps[k]);
            ba->split_bitmaps[k] = NULL;
        }
    }
}

void
buddy_allocator_t::_lazy_init(int area_idx)
{
    assert(area_idx >= 0);
    assert(area_idx < this->_nmemories);

    buddy_area_t * ba = &(this->_buddy_areas[area_idx]);

    if (ba->alloc_size > 0)
        return ;

    XKRT_MUTEX_LOCK(ba->lock);
    {
        if (ba->alloc_size == 0)
        {
            size_t size = _compute_size(this->_memory_size_initial, this->_capacities[area_idx]);
            assert(this->_f_alloc);

            /* try to allocate device memory, halving the size on failure */
            void * device_ptr = NULL;
            while (size > 0)
            {
                device_ptr = this->_f_alloc(this->_device_driver_id, size, area_idx);
                if (device_ptr != NULL)
                    break ;
                size >>= 1;
            }
            if (device_ptr == NULL)
                LOGGER_FATAL("Out of GPU memory");

            this->_init_area(area_idx, (uintptr_t) device_ptr, size);
        }
    }
    XKRT_MUTEX_UNLOCK(ba->lock);
}

///////////////////////////
// ALLOCATE              //
///////////////////////////

area_chunk_t *
buddy_allocator_t::allocate_on(const size_t user_size, int area_idx)
{
    /* ensure backing device memory is allocated for this area */
    this->_lazy_init(area_idx);

    buddy_area_t * ba = &(this->_buddy_areas[area_idx]);

    /* align data */
    const size_t size = (user_size + 7UL) & ~7UL;

    /* find the order needed */
    const int order = _size_to_order(size);
    if (order >= ba->norders)
        return NULL; /* requested size exceeds pool */

    area_chunk_t * chunk = NULL;

    XKRT_MUTEX_LOCK(ba->lock);
    {
        /* find the smallest order >= requested order that has a free block */
        int k = order;
        while (k < ba->norders && ba->free_lists[k] == NULL)
            ++k;

        if (k < ba->norders)
        {
            /* pop a block from the free list at order k */
            chunk = ba->free_lists[k];
            ba->free_lists[k] = chunk->freelink;
            chunk->freelink = NULL;

            /* split down to the requested order */
            while (k > order)
            {
                --k;
                size_t half_size = BUDDY_MIN_BLOCK_SIZE << k;

                /* create the buddy (right half) */
                area_chunk_t * buddy = (area_chunk_t *) malloc(sizeof(area_chunk_t));
                assert(buddy);
                buddy->ptr         = chunk->ptr + half_size;
                buddy->size        = half_size;
                buddy->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
                buddy->prev        = NULL;
                buddy->next        = NULL;
                buddy->freelink    = ba->free_lists[k];
                buddy->use_counter = 0;
                buddy->area_idx    = area_idx;

                /* put the buddy on the free list at order k */
                ba->free_lists[k] = buddy;

                /* shrink current chunk to the left half */
                chunk->size = half_size;

                /* toggle split bitmap at order k:
                 * the buddy pair index = offset_of_left_block / (2 * half_size)
                 *                      = (chunk->ptr - base_ptr) / (2 * half_size) */
                if (ba->split_bitmaps[k])
                {
                    size_t pair_idx = (chunk->ptr - ba->base_ptr) / (half_size * 2);
                    _bitmap_toggle(ba->split_bitmaps[k], pair_idx);
                }
            }

            /* mark the block as allocated */
            chunk->state       = XKRT_ALLOC_CHUNK_STATE_ALLOCATED;
            chunk->area_idx    = area_idx;
            chunk->use_counter = 0;
            chunk->prev        = NULL;
            chunk->next        = NULL;
            chunk->freelink    = NULL;

            /* toggle split bitmap at the allocation order:
             * this marks that one buddy of the pair is now allocated */
            if (order < ba->norders - 1 && ba->split_bitmaps[order])
            {
                size_t block_size = BUDDY_MIN_BLOCK_SIZE << order;
                size_t pair_idx = (chunk->ptr - ba->base_ptr) / (block_size * 2);
                _bitmap_toggle(ba->split_bitmaps[order], pair_idx);
            }
        }
    }
    XKRT_MUTEX_UNLOCK(ba->lock);

    return chunk;
}

area_chunk_t *
buddy_allocator_t::allocate(const size_t user_size)
{
    return this->allocate_on(user_size, 0);
}

///////////////////////////
// DEALLOCATE            //
///////////////////////////

void
buddy_allocator_t::deallocate(area_chunk_t * chunk)
{
    return this->deallocate_on(chunk, chunk->area_idx);
}

void
buddy_allocator_t::deallocate_on(area_chunk_t * chunk, int area_idx)
{
    assert(chunk);
    assert(chunk->area_idx >= 0);

    buddy_area_t * ba = &(this->_buddy_areas[area_idx]);

    XKRT_MUTEX_LOCK(ba->lock);
    {
        chunk->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
        chunk->use_counter = 0;

        int order = _size_to_order(chunk->size);
        assert(order < ba->norders);
        assert(chunk->size == (BUDDY_MIN_BLOCK_SIZE << order));

        /* try to coalesce with buddy */
        while (order < ba->norders - 1)
        {
            size_t block_size = BUDDY_MIN_BLOCK_SIZE << order;
            size_t offset     = chunk->ptr - ba->base_ptr;

            /* buddy address: XOR with block_size flips the relevant bit */
            uintptr_t buddy_ptr = ba->base_ptr + (offset ^ block_size);

            /* toggle the split bitmap to check if buddy is also free */
            size_t pair_idx = offset / (block_size * 2);
            /* After toggle: 0 means both buddies are free (can coalesce),
             *               1 means only one is free (cannot coalesce) */
            int bit = _bitmap_toggle(ba->split_bitmaps[order], pair_idx);

            if (bit != 0)
            {
                /* buddy is still allocated, cannot coalesce */
                break ;
            }

            /* buddy is free — find and remove it from the free list at this order */
            area_chunk_t * buddy = NULL;
            {
                area_chunk_t * prev = NULL;
                area_chunk_t * curr = ba->free_lists[order];
                while (curr)
                {
                    if (curr->ptr == buddy_ptr)
                    {
                        buddy = curr;
                        if (prev)
                            prev->freelink = curr->freelink;
                        else
                            ba->free_lists[order] = curr->freelink;
                        curr->freelink = NULL;
                        break ;
                    }
                    prev = curr;
                    curr = curr->freelink;
                }
            }

            if (buddy == NULL)
            {
                /* buddy not found in free list — should not happen if bitmap
                 * was consistent; re-toggle to restore and stop */
                _bitmap_toggle(ba->split_bitmaps[order], pair_idx);
                break ;
            }

            /* coalesce: keep the lower-address block, free the other */
            if (chunk->ptr < buddy->ptr)
            {
                chunk->size = block_size * 2;
                free(buddy);
            }
            else
            {
                buddy->size = block_size * 2;
                free(chunk);
                chunk = buddy;
            }

            ++order;
        }

        /* add the (possibly coalesced) block to the free list */
        chunk->freelink    = ba->free_lists[order];
        ba->free_lists[order] = chunk;
        chunk->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
        chunk->use_counter = 0;
    }
    XKRT_MUTEX_UNLOCK(ba->lock);
}

///////////////////////////
// RESET / DESTRUCTOR    //
///////////////////////////

void
buddy_allocator_t::reset_on(int area_idx)
{
    assert(area_idx >= 0);
    assert(area_idx < this->_nmemories);

    buddy_area_t * ba = &(this->_buddy_areas[area_idx]);

    if (ba->alloc_size == 0)
        return ;

    /* free all chunks in all free lists */
    for (int k = 0 ; k < ba->norders ; ++k)
    {
        area_chunk_t * curr = ba->free_lists[k];
        while (curr)
        {
            area_chunk_t * next = curr->freelink;
            free(curr);
            curr = next;
        }
        ba->free_lists[k] = NULL;
    }

    /* free bitmaps */
    this->_free_bitmaps(area_idx);

    /* deallocate backing device memory */
    assert(this->_f_dealloc);
    this->_f_dealloc(this->_device_driver_id, (void *) ba->base_ptr, ba->alloc_size, area_idx);

    /* reset to uninitialized — next allocate_on will re-allocate */
    ba->base_ptr   = 0;
    ba->alloc_size = 0;
    ba->pool_size  = 0;
    ba->norders    = 0;
}

void
buddy_allocator_t::reset(void)
{
    for (int i = 0 ; i < this->_nmemories ; ++i)
        this->reset_on(i);
}

buddy_allocator_t::~buddy_allocator_t()
{
    for (int i = 0 ; i < this->_nmemories ; ++i)
    {
        buddy_area_t * ba = &(this->_buddy_areas[i]);

        if (ba->alloc_size == 0)
            continue ;

        /* free all chunks in all free lists */
        for (int k = 0 ; k < ba->norders ; ++k)
        {
            area_chunk_t * curr = ba->free_lists[k];
            while (curr)
            {
                area_chunk_t * next = curr->freelink;
                free(curr);
                curr = next;
            }
            ba->free_lists[k] = NULL;
        }

        /* free bitmaps */
        this->_free_bitmaps(i);

        /* deallocate backing device memory */
        assert(this->_f_dealloc);
        this->_f_dealloc(this->_device_driver_id, (void *) ba->base_ptr, ba->alloc_size, i);

        ba->base_ptr   = 0;
        ba->alloc_size = 0;
        ba->pool_size  = 0;
        ba->norders    = 0;
    }
}
