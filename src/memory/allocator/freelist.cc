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

# include <xkrt/memory/allocator/freelist.hpp>
# include <xkrt/logger/logger.h>

# include <cassert>
# include <cstring>
# include <cstdlib>

XKRT_NAMESPACE_USE;

freelist_allocator_t::freelist_allocator_t(
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
    memset(this->_areas, 0, sizeof(this->_areas));
    /* _backing[] elements are default-constructed by small_vector_t (no memset needed) */
    for (int i = 0 ; i < XKRT_DEVICE_MEMORIES_MAX ; ++i)
        XKRT_MUTEX_INIT(this->_areas[i].lock);
}

bool
freelist_allocator_t::_add_backing_region(int area_idx)
{
    area_t * area = &(this->_areas[area_idx]);

    /* determine size: first allocation uses memory_size_initial,
     * subsequent ones use memory_size_resize */
    const memory_size_t & ms = (this->_backing[area_idx].empty())
        ? this->_memory_size_initial
        : this->_memory_size_resize;

    size_t size = memory_size_compute(ms, this->_capacities[area_idx]);
    if (size == 0)
        return false;

    /* try to allocate device memory, halving the size on failure */
    assert(this->_f_alloc);
    void * device_ptr = NULL;
    while (size > 0)
    {
        device_ptr = this->_f_alloc(this->_device_driver_id, size, area_idx);
        if (device_ptr != NULL)
            break ;
        size >>= 1;
    }
    if (device_ptr == NULL)
        return false;

    /* create a free chunk covering the entire region */
    area_chunk_t * chunk = (area_chunk_t *) malloc(sizeof(area_chunk_t));
    assert(chunk);
    chunk->ptr         = (uintptr_t) device_ptr;
    chunk->size        = size;
    chunk->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
    chunk->prev        = NULL;
    chunk->next        = NULL;
    chunk->freelink    = area->free_chunk_list;
    chunk->use_counter = 0;
    chunk->area_idx    = area_idx;

    area->free_chunk_list = chunk;

    /* track the backing region — small_vector grows automatically */
    backing_region_t * region = this->_backing[area_idx].emplace_back();
    region->ptr  = (uintptr_t) device_ptr;
    region->size = size;

    return true;
}

void
freelist_allocator_t::_free_area_chunks(int area_idx)
{
    area_t * area = &(this->_areas[area_idx]);

    /*
     * We need to free ALL area_chunk_t structs (both free and allocated).
     * The free_chunk_list only links free chunks via 'freelink'.
     * However, each chunk also lives in an ordered doubly-linked list
     * (via prev/next) that covers all chunks within its backing region.
     *
     * Strategy:
     * 1. Walk the free list and collect unique chain heads (prev == NULL).
     * 2. For each unique head, walk the ordered 'next' chain and free
     *    every chunk.
     *
     * Note: if all chunks in a backing region are currently allocated
     * (none free), those chunk structs will not be reached here and
     * will be leaked.  In practice this does not happen because
     * reset/finalize is called after the memory-tree has freed
     * everything.
     */

    /* step 1: collect unique chain heads (inline storage avoids heap in the common case) */
    small_vector_t<area_chunk_t *, 4> heads;

    for (area_chunk_t * fchunk = area->free_chunk_list ; fchunk ; fchunk = fchunk->freelink)
    {
        /* find the head of this ordered chain */
        area_chunk_t * head = fchunk;
        while (head->prev)
            head = head->prev;

        /* check for duplicates */
        bool found = false;
        for (int i = 0 ; i < heads.size() ; ++i)
        {
            if (heads[i] == head)
            {
                found = true;
                break ;
            }
        }
        if (!found)
            heads.push_back(head);
    }

    /* step 2: walk each chain and free every chunk */
    for (int i = 0 ; i < heads.size() ; ++i)
    {
        area_chunk_t * curr = heads[i];
        while (curr)
        {
            area_chunk_t * next = curr->next;
            free(curr);
            curr = next;
        }
    }
    /* heads destructor frees overflow buffer automatically */
    area->free_chunk_list = NULL;
}

void
freelist_allocator_t::_lazy_init(int area_idx)
{
    assert(area_idx >= 0);
    assert(area_idx < this->_nmemories);

    if (!this->_backing[area_idx].empty())
        return ;

    area_t * area = &(this->_areas[area_idx]);

    XKRT_MUTEX_LOCK(area->lock);
    {
        if (this->_backing[area_idx].empty())
        {
            if (!this->_add_backing_region(area_idx))
                LOGGER_FATAL("Out of GPU memory");
        }
    }
    XKRT_MUTEX_UNLOCK(area->lock);
}

void
freelist_allocator_t::reset_on(int area_idx)
{
    assert(area_idx >= 0);
    assert(area_idx < this->_nmemories);

    if (this->_backing[area_idx].empty())
        return ;

    area_t * area = &(this->_areas[area_idx]);

    XKRT_MUTEX_LOCK(area->lock);
    {
        /* free all chunk metadata nodes */
        this->_free_area_chunks(area_idx);

        assert(this->_f_dealloc);

        /* save the initial (first) backing region so it can be recycled */
        backing_region_t initial = this->_backing[area_idx][0];

        /* release all additional backing regions back to the driver */
        for (int j = 1 ; j < this->_backing[area_idx].size() ; ++j)
        {
            backing_region_t * r = &(this->_backing[area_idx][j]);
            this->_f_dealloc(this->_device_driver_id, (void *) r->ptr, r->size, area_idx);
        }

        /* shrink the backing list back to just the initial region */
        this->_backing[area_idx].clear();
        this->_backing[area_idx].push_back(initial);

        /* reinstall a single free chunk covering the recycled initial region */
        area_chunk_t * chunk = (area_chunk_t *) malloc(sizeof(area_chunk_t));
        assert(chunk);
        chunk->ptr         = initial.ptr;
        chunk->size        = initial.size;
        chunk->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
        chunk->prev        = NULL;
        chunk->next        = NULL;
        chunk->freelink    = NULL;
        chunk->use_counter = 0;
        chunk->area_idx    = area_idx;
        area->free_chunk_list = chunk;
    }
    XKRT_MUTEX_UNLOCK(area->lock);
}

void
freelist_allocator_t::reset(void)
{
    for (int i = 0 ; i < this->_nmemories ; ++i)
        this->reset_on(i);
}

freelist_allocator_t::~freelist_allocator_t()
{
    for (int i = 0 ; i < this->_nmemories ; ++i)
    {
        if (this->_backing[i].empty())
            continue ;

        /* free all chunk metadata */
        this->_free_area_chunks(i);

        /* deallocate all backing device memory regions */
        assert(this->_f_dealloc);
        for (int j = 0 ; j < this->_backing[i].size() ; ++j)
        {
            backing_region_t * r = &(this->_backing[i][j]);
            this->_f_dealloc(this->_device_driver_id, (void *) r->ptr, r->size, i);
        }
        /* _backing[i] destructor releases the overflow buffer automatically */
    }
}

void
freelist_allocator_t::deallocate(area_chunk_t * chunk)
{
    return this->deallocate_on(chunk, chunk->area_idx);
}

void
freelist_allocator_t::deallocate_on(area_chunk_t * chunk, int area_idx)
{
    assert(chunk->area_idx >= 0);
    area_t * area = &(this->_areas[area_idx]);

    bool delete_chunk = false;
    XKRT_MUTEX_LOCK(area->lock);
    {
        chunk->state = XKRT_ALLOC_CHUNK_STATE_FREE;
        chunk->use_counter = 0;

        /* can we merge chunk into next_chunk ? */
        area_chunk_t * next_chunk = chunk->next;
        if (next_chunk && next_chunk->state == XKRT_ALLOC_CHUNK_STATE_FREE)
        {
            next_chunk->prev = chunk->prev;
            if (chunk->prev)
                chunk->prev->next = next_chunk;
            next_chunk->size += chunk->size;
            assert(next_chunk->ptr > chunk->ptr);
            next_chunk->ptr = chunk->ptr;
            delete_chunk = true;
        }

        area_chunk_t * prev_chunk = chunk->prev;
        if (prev_chunk)
        {
            /*  if prev_chunk is a free chunk and 'delete_chunk' is true,
             *  then we have to merge prev and next */
            if (prev_chunk->state == XKRT_ALLOC_CHUNK_STATE_FREE)
            {
                if (delete_chunk)
                {
                    assert(prev_chunk->ptr < chunk->ptr);
                    assert(prev_chunk->ptr < next_chunk->ptr);

                    prev_chunk->size += next_chunk->size;
                    prev_chunk->next = next_chunk->next;
                    if (next_chunk->next)
                        next_chunk->next->prev = prev_chunk;
                    prev_chunk->freelink = next_chunk->freelink;
                    free(next_chunk);
                }
                else
                {
                    /* merge chunk into prev_chunk */
                    assert(prev_chunk->ptr < chunk->ptr);
                    prev_chunk->next = chunk->next;
                    if (chunk->next)
                        chunk->next->prev = prev_chunk;
                    prev_chunk->size += chunk->size;
                    delete_chunk = true;
                }
            }
            else if (!delete_chunk)
            {
                /* free_chunk_list is ordered by increasing adress: search form prev the previous bloc */
                while (prev_chunk && prev_chunk->state != XKRT_ALLOC_CHUNK_STATE_FREE)
                    prev_chunk = prev_chunk->prev;

                if (!prev_chunk)
                {
                    chunk->freelink = area->free_chunk_list;
                    area->free_chunk_list = chunk;
                }
                else
                {
                    chunk->freelink = prev_chunk->freelink;
                    prev_chunk->freelink = chunk;
                }
            }
        }
        else if (!delete_chunk)
        {
            chunk->freelink = area->free_chunk_list;
            area->free_chunk_list = chunk;
        }
    }
    XKRT_MUTEX_UNLOCK(area->lock);

    if (delete_chunk)
        free(chunk);
}

area_chunk_t *
freelist_allocator_t::allocate_on(const size_t user_size, int area_idx)
{
    /* ensure backing device memory is allocated for this area */
    this->_lazy_init(area_idx);

    area_t * area = &(this->_areas[area_idx]);

    /* align data */
    const size_t size = (user_size + 7UL) & ~7UL;
    area_chunk_t * curr;

    XKRT_MUTEX_LOCK(area->lock);
    {
        /* best fit strategy */
        curr = area->free_chunk_list;

        area_chunk_t * prevfree = NULL;
        size_t min_size = 0;
        area_chunk_t * min_size_curr = NULL;
        area_chunk_t * min_size_prevfree = NULL;

        while (curr)
        {
            size_t curr_size = curr->size;
            if (curr_size >= size)
            {
                if ((min_size_curr == 0) || (min_size > curr_size))
                {
                    min_size = curr_size;
                    min_size_curr = curr;
                    min_size_prevfree = prevfree;
                }
            }
            prevfree = curr;
            curr = curr->freelink;
        }

        /* and the winner is min_size_curr ! */
        curr = min_size_curr;
        prevfree = min_size_prevfree;

        /* split chunk */
        if ((curr != NULL) && (min_size - size >= size / 2))
        {
            size_t curr_size = curr->size;
            area_chunk_t * remainder = (area_chunk_t *) malloc(sizeof(area_chunk_t));
            remainder->ptr         = size + curr->ptr;
            remainder->size        = (curr_size - size);
            remainder->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
            remainder->use_counter = 0;
            remainder->prev        = curr;
            remainder->next        = curr->next;
            remainder->freelink    = curr->freelink;

            /* link remainder segment after curr */
            if (curr->next)
                curr->next->prev = remainder;
            curr->next = remainder;
            curr->size = size;
            curr->freelink = remainder;
        }

        if (curr != NULL)
        {
            if (prevfree)
                prevfree->freelink = curr->freelink;
            else
                area->free_chunk_list = curr->freelink;
            curr->state = XKRT_ALLOC_CHUNK_STATE_ALLOCATED;
            curr->freelink = NULL;
        }
    }

    XKRT_MUTEX_UNLOCK(area->lock);

    if (curr)
    {
        curr->area_idx = area_idx;
    }

    return curr;
}

area_chunk_t *
freelist_allocator_t::allocate(const size_t user_size)
{
    return this->allocate_on(user_size, 0);
}
