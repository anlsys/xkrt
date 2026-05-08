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

#ifndef __XKRT_MEMORY_ALLOCATOR_HPP__
# define __XKRT_MEMORY_ALLOCATOR_HPP__

# include <xkrt/types.h>
# include <xkrt/namespace.h>

# include <stddef.h>
# include <stdint.h>

XKRT_NAMESPACE_BEGIN

/**
 *  Function pointer types for driver-level memory allocation/deallocation.
 *  These match the signatures in driver_t (driver.h:143-144).
 */
typedef void * (*f_memory_device_allocate_t)(device_driver_id_t device_driver_id, const size_t size, int area_idx);
typedef void   (*f_memory_device_deallocate_t)(device_driver_id_t device_driver_id, void * ptr, const size_t size, int area_idx);

/**
 *  Abstract allocator interface.
 *
 *  An allocator manages a pool of device memory, subdivided into areas.
 *  It lazily allocates backing device memory on first use per area,
 *  using driver-provided function pointers.
 *
 *  The allocator owns the lifecycle of the backing device memory:
 *    - On first allocate_on() for a given area, it calls f_alloc to get
 *      backing memory and sets up the area internally.
 *    - On reset(), it deallocates all backing device memory and returns
 *      to initial state (no memory allocated).
 *    - On deinitialize(), it deallocates all backing device memory (for teardown).
 */
class allocator_t
{
    protected:
        /* size configuration */
        memory_size_t _memory_size_initial;
        memory_size_t _memory_size_resize;

        /* driver callbacks for backing memory */
        f_memory_device_allocate_t   _f_alloc;
        f_memory_device_deallocate_t _f_dealloc;

        /* driver device id */
        device_driver_id_t _device_driver_id;

        /* number of memory areas and their capacities */
        int    _nmemories;
        size_t _capacities[XKRT_DEVICE_MEMORIES_MAX];

    public:
        allocator_t(
            memory_size_t                   memory_size_initial,
            memory_size_t                   memory_size_resize,
            f_memory_device_allocate_t      f_alloc,
            f_memory_device_deallocate_t    f_dealloc,
            device_driver_id_t              device_driver_id,
            int                             nmemories,
            const size_t                  * capacities
        )
            : _memory_size_initial(memory_size_initial)
            , _memory_size_resize(memory_size_resize)
            , _f_alloc(f_alloc)
            , _f_dealloc(f_dealloc)
            , _device_driver_id(device_driver_id)
            , _nmemories(nmemories)
        {
            for (int i = 0 ; i < nmemories ; ++i)
                this->_capacities[i] = capacities[i];
        }

        virtual ~allocator_t() {}

        /**
         *  Allocate a chunk of at least 'size' bytes from the given area.
         *  On first call for a given area, lazily allocates backing device
         *  memory via f_alloc.
         *  Returns the allocated chunk, or NULL if no suitable chunk is found.
         */
        virtual area_chunk_t * allocate_on(const size_t size, int area_idx) = 0;

        /**
         *  Allocate a chunk of at least 'size' bytes from area 0.
         */
        virtual area_chunk_t * allocate(const size_t size) = 0;

        /**
         *  Deallocate a chunk from the given area,
         *  returning it to the free pool.
         */
        virtual void deallocate_on(area_chunk_t * chunk, int area_idx) = 0;

        /**
         *  Deallocate a chunk (area is inferred from the chunk).
         */
        virtual void deallocate(area_chunk_t * chunk) = 0;

        /**
         *  Reset all areas managed by this allocator.
         *  Deallocates backing device memory via f_dealloc and
         *  returns to initial state with no memory allocated.
         */
        virtual void reset(void) = 0;

        /**
         *  Reset the given area, deallocating its backing device memory
         *  and returning to initial state.
         */
        virtual void reset_on(int area_idx) = 0;
};

XKRT_NAMESPACE_END

#endif /* __XKRT_MEMORY_ALLOCATOR_HPP__ */
