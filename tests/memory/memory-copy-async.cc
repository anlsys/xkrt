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
 *  Exercises runtime_t::memory_copy_async with a host<->device round-trip:
 *      host(src) --H2D--> device --D2H--> host(dst)
 *  and verifies that dst == src.
 *
 *  This requires a real (non-host) device and is skipped gracefully otherwise.
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <common/skip.h>

# include <assert.h>
# include <stdlib.h>
# include <string.h>

XKRT_NAMESPACE_USE;

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    // memory_copy_async submits copies to a (non-host) device's queues
    XKRT_TEST_SKIP_IF_NO_GPU(runtime);

    const device_unique_id_t gpu     = 1;       // first non-host device
    const size_t             size    = 1 << 20; // 1 MiB
    const int                nchunks = 8;

    // host buffers (registered for DMA)
    unsigned char * src = (unsigned char *) malloc(size);
    unsigned char * dst = (unsigned char *) malloc(size);
    assert(src && dst);
    for (size_t i = 0 ; i < size ; ++i)
        src[i] = (unsigned char) ((i * 31 + 7) % 256);
    memset(dst, 0, size);

    assert(runtime.memory_register(src, size) == 0);
    assert(runtime.memory_register(dst, size) == 0);

    // device scratch buffer
    area_chunk_t * chunk = runtime.memory_device_allocate(gpu, size);
    assert(chunk != NULL);
    assert(chunk->size >= size);

    // H2D: host src -> device
    runtime.memory_copy_async(
        gpu, size,
        /* dst */ gpu,                        chunk->ptr,
        /* src */ XKRT_HOST_DEVICE_UNIQUE_ID, (uintptr_t) src,
        nchunks);
    runtime.task_wait();

    // D2H: device -> host dst
    runtime.memory_copy_async(
        gpu, size,
        /* dst */ XKRT_HOST_DEVICE_UNIQUE_ID, (uintptr_t) dst,
        /* src */ gpu,                        chunk->ptr,
        nchunks);
    runtime.task_wait();

    // verify the round-trip
    for (size_t i = 0 ; i < size ; ++i)
        assert(dst[i] == src[i]);

    runtime.memory_device_deallocate(gpu, chunk);
    runtime.memory_unregister(src, size);
    runtime.memory_unregister(dst, size);
    free(src);
    free(dst);

    assert(runtime.deinit() == 0);

    return 0;
}
