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

/**
 *  Unit test for memory allocators (freelist and buddy).
 *
 *  Uses the runtime_t allocation methods to exercise allocate, deallocate,
 *  and deallocate_all on the host device (device 0).
 *
 *  The allocator type is determined by the XKRT_ALLOCATOR_TYPE environment
 *  variable ("freelist" by default, or "buddy").
 *
 *  This test exercises:
 *    1. Single allocation and deallocation
 *    2. Multiple allocations of varying sizes
 *    3. Deallocation in different orders (LIFO, FIFO, random)
 *    4. Reallocation after deallocation (verifying freed space is reusable)
 *    5. deallocate_all and re-allocation
 *    6. Many small allocations (stress test)
 *    7. Alignment (all returned pointers must be 8-byte aligned)
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <stdlib.h>
# include <string.h>

XKRT_NAMESPACE_USE;

/* host device is always device 0 */
static constexpr device_unique_id_t HOST_DEVICE = XKRT_HOST_DEVICE_UNIQUE_ID;

/**
 *  Test 1: Single allocation and deallocation
 */
static void
test_single_alloc_dealloc(runtime_t * runtime)
{
    LOGGER_INFO("  test_single_alloc_dealloc");

    area_chunk_t * chunk = runtime->memory_device_allocate(HOST_DEVICE, 1024);
    assert(chunk != NULL);
    assert(chunk->size >= 1024);
    assert(chunk->ptr != 0);
    assert((chunk->ptr & 7) == 0); /* 8-byte aligned */

    runtime->memory_device_deallocate(HOST_DEVICE, chunk);
}

/**
 *  Test 2: Multiple allocations of varying sizes
 */
static void
test_multiple_alloc(runtime_t * runtime)
{
    LOGGER_INFO("  test_multiple_alloc");

    static const size_t sizes[] = { 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 8192, 65536 };
    static const int nsizes = (int) (sizeof(sizes) / sizeof(sizes[0]));

    area_chunk_t * chunks[nsizes];

    /* allocate all */
    for (int i = 0 ; i < nsizes ; ++i)
    {
        chunks[i] = runtime->memory_device_allocate(HOST_DEVICE, sizes[i]);
        assert(chunks[i] != NULL);
        assert(chunks[i]->size >= sizes[i]);
        assert(chunks[i]->ptr != 0);
        assert((chunks[i]->ptr & 7) == 0);
    }

    /* verify no overlaps: all chunk [ptr, ptr+size) ranges are disjoint */
    for (int i = 0 ; i < nsizes ; ++i)
    {
        for (int j = i + 1 ; j < nsizes ; ++j)
        {
            uintptr_t a_start = chunks[i]->ptr;
            uintptr_t a_end   = chunks[i]->ptr + chunks[i]->size;
            uintptr_t b_start = chunks[j]->ptr;
            uintptr_t b_end   = chunks[j]->ptr + chunks[j]->size;
            assert(a_end <= b_start || b_end <= a_start);
        }
    }

    /* deallocate all */
    for (int i = 0 ; i < nsizes ; ++i)
        runtime->memory_device_deallocate(HOST_DEVICE, chunks[i]);
}

/**
 *  Test 3: Deallocation in different orders
 */
static void
test_dealloc_orders(runtime_t * runtime)
{
    LOGGER_INFO("  test_dealloc_orders");

    /* LIFO order */
    {
        static const int N = 8;
        area_chunk_t * chunks[N];
        for (int i = 0 ; i < N ; ++i)
        {
            chunks[i] = runtime->memory_device_allocate(HOST_DEVICE, 512);
            assert(chunks[i] != NULL);
        }
        for (int i = N - 1 ; i >= 0 ; --i)
            runtime->memory_device_deallocate(HOST_DEVICE, chunks[i]);
    }

    /* FIFO order */
    {
        static const int N = 8;
        area_chunk_t * chunks[N];
        for (int i = 0 ; i < N ; ++i)
        {
            chunks[i] = runtime->memory_device_allocate(HOST_DEVICE, 512);
            assert(chunks[i] != NULL);
        }
        for (int i = 0 ; i < N ; ++i)
            runtime->memory_device_deallocate(HOST_DEVICE, chunks[i]);
    }

    /* interleaved: free even indices, then odd */
    {
        static const int N = 8;
        area_chunk_t * chunks[N];
        for (int i = 0 ; i < N ; ++i)
        {
            chunks[i] = runtime->memory_device_allocate(HOST_DEVICE, 512);
            assert(chunks[i] != NULL);
        }
        for (int i = 0 ; i < N ; i += 2)
            runtime->memory_device_deallocate(HOST_DEVICE, chunks[i]);
        for (int i = 1 ; i < N ; i += 2)
            runtime->memory_device_deallocate(HOST_DEVICE, chunks[i]);
    }
}

/**
 *  Test 4: Reallocation after deallocation
 *  Verifies that freed space can be reused.
 */
static void
test_realloc_after_free(runtime_t * runtime)
{
    LOGGER_INFO("  test_realloc_after_free");

    /* allocate a large chunk, free it, then allocate again */
    area_chunk_t * chunk1 = runtime->memory_device_allocate(HOST_DEVICE, 16384);
    assert(chunk1 != NULL);

    runtime->memory_device_deallocate(HOST_DEVICE, chunk1);

    /* should be able to re-allocate the same size */
    area_chunk_t * chunk2 = runtime->memory_device_allocate(HOST_DEVICE, 16384);
    assert(chunk2 != NULL);
    assert(chunk2->size >= 16384);

    runtime->memory_device_deallocate(HOST_DEVICE, chunk2);

    /* allocate-free-allocate cycle with mixed sizes */
    area_chunk_t * a = runtime->memory_device_allocate(HOST_DEVICE, 4096);
    area_chunk_t * b = runtime->memory_device_allocate(HOST_DEVICE, 8192);
    assert(a != NULL);
    assert(b != NULL);

    runtime->memory_device_deallocate(HOST_DEVICE, a);

    /* allocate something that fits in the freed 'a' space */
    area_chunk_t * c = runtime->memory_device_allocate(HOST_DEVICE, 2048);
    assert(c != NULL);

    runtime->memory_device_deallocate(HOST_DEVICE, b);
    runtime->memory_device_deallocate(HOST_DEVICE, c);
}

/**
 *  Test 5: deallocate_all and re-allocation
 */
static void
test_deallocate_all(runtime_t * runtime)
{
    LOGGER_INFO("  test_deallocate_all");

    /* allocate several chunks */
    static const int N = 16;
    for (int i = 0 ; i < N ; ++i)
    {
        area_chunk_t * chunk = runtime->memory_device_allocate(HOST_DEVICE, 1024);
        assert(chunk != NULL);
    }

    /* deallocate all at once */
    runtime->memory_device_deallocate_all(HOST_DEVICE);

    /* should be able to allocate again after reset */
    area_chunk_t * chunk = runtime->memory_device_allocate(HOST_DEVICE, 4096);
    assert(chunk != NULL);
    assert(chunk->size >= 4096);

    runtime->memory_device_deallocate(HOST_DEVICE, chunk);
}

/**
 *  Test 6: Many small allocations (stress test)
 */
static void
test_many_small_allocs(runtime_t * runtime)
{
    LOGGER_INFO("  test_many_small_allocs");

    static const int N = 256;
    area_chunk_t * chunks[N];

    for (int i = 0 ; i < N ; ++i)
    {
        chunks[i] = runtime->memory_device_allocate(HOST_DEVICE, 256);
        assert(chunks[i] != NULL);
        assert(chunks[i]->size >= 256);
        assert((chunks[i]->ptr & 7) == 0);
    }

    /* verify no overlaps */
    for (int i = 0 ; i < N ; ++i)
    {
        for (int j = i + 1 ; j < N ; ++j)
        {
            uintptr_t a_end   = chunks[i]->ptr + chunks[i]->size;
            uintptr_t b_start = chunks[j]->ptr;
            uintptr_t b_end   = chunks[j]->ptr + chunks[j]->size;
            uintptr_t a_start = chunks[i]->ptr;
            assert(a_end <= b_start || b_end <= a_start);
        }
    }

    /* free all */
    for (int i = 0 ; i < N ; ++i)
        runtime->memory_device_deallocate(HOST_DEVICE, chunks[i]);
}

/**
 *  Test 7: Allocate-free-allocate cycles to test coalescing/buddy merging
 */
static void
test_coalescing(runtime_t * runtime)
{
    LOGGER_INFO("  test_coalescing");

    /* allocate two adjacent chunks, free both, then allocate
     * a larger chunk that requires the freed space to be merged */
    area_chunk_t * a = runtime->memory_device_allocate(HOST_DEVICE, 4096);
    area_chunk_t * b = runtime->memory_device_allocate(HOST_DEVICE, 4096);
    assert(a != NULL);
    assert(b != NULL);

    runtime->memory_device_deallocate(HOST_DEVICE, a);
    runtime->memory_device_deallocate(HOST_DEVICE, b);

    /* after freeing both, the allocator should be able to merge them
     * and serve a larger allocation */
    area_chunk_t * c = runtime->memory_device_allocate(HOST_DEVICE, 8192);
    assert(c != NULL);
    assert(c->size >= 8192);

    runtime->memory_device_deallocate(HOST_DEVICE, c);
}

/**
 *  Test 8: Edge case — very small allocation (1 byte)
 */
static void
test_tiny_alloc(runtime_t * runtime)
{
    LOGGER_INFO("  test_tiny_alloc");

    area_chunk_t * chunk = runtime->memory_device_allocate(HOST_DEVICE, 1);
    assert(chunk != NULL);
    assert(chunk->size >= 1);
    assert((chunk->ptr & 7) == 0);

    runtime->memory_device_deallocate(HOST_DEVICE, chunk);
}

/**
 *  Run all allocator tests with the currently configured allocator.
 */
static void
run_allocator_tests(const char * allocator_name)
{
    LOGGER_INFO("Running allocator tests with '%s'", allocator_name);

    runtime_t runtime;
    assert(runtime.init() == 0);

    test_single_alloc_dealloc(&runtime);
    test_multiple_alloc(&runtime);
    test_dealloc_orders(&runtime);
    test_realloc_after_free(&runtime);
    test_deallocate_all(&runtime);
    test_many_small_allocs(&runtime);
    test_coalescing(&runtime);
    test_tiny_alloc(&runtime);

    assert(runtime.deinit() == 0);

    LOGGER_INFO("All '%s' allocator tests passed", allocator_name);
}

int
main(void)
{
    /* test with freelist allocator */
    setenv("XKRT_ALLOCATOR_TYPE", "freelist", 1);
    run_allocator_tests("freelist");

    /* test with buddy allocator */
    setenv("XKRT_ALLOCATOR_TYPE", "buddy", 1);
    run_allocator_tests("buddy");

    return 0;
}
