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
 *  Exercises the host and unified memory allocation API of runtime_t
 *  (memory_host_allocate / memory_unified_allocate), complementing
 *  memory-allocator.cc which only covers the device (area) allocator.
 *
 *  - Host allocation always works (the host driver falls back to malloc).
 *  - Unified allocation is a *device* capability: it is only attempted when a
 *    non-host device is present (otherwise the runtime would abort). This test
 *    is therefore host-runnable and additionally exercises unified memory on
 *    GPU builds.
 */

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>
# include <string.h>

XKRT_NAMESPACE_USE;

static constexpr device_unique_id_t HOST_DEVICE = XKRT_HOST_DEVICE_UNIQUE_ID;

static void
test_host_allocate(runtime_t & runtime)
{
    LOGGER_INFO("  test_host_allocate");

    static const size_t sizes[] = { 1, 8, 64, 4096, 1 << 20 };
    for (size_t s = 0 ; s < sizeof(sizes) / sizeof(sizes[0]) ; ++s)
    {
        const size_t size = sizes[s];

        unsigned char * mem = (unsigned char *) runtime.memory_host_allocate(HOST_DEVICE, size);
        assert(mem != NULL);

        // memory must be writable / readable
        for (size_t i = 0 ; i < size ; ++i)
            mem[i] = (unsigned char) (i % 256);
        for (size_t i = 0 ; i < size ; ++i)
            assert(mem[i] == (unsigned char) (i % 256));

        runtime.memory_host_deallocate(HOST_DEVICE, mem, size);
    }
}

static void
test_unified_allocate(runtime_t & runtime)
{
    // unified memory is only valid on a non-host (GPU) device
    if (runtime.get_ndevices() <= 1)
    {
        LOGGER_WARN("  test_unified_allocate skipped: no GPU device present");
        return;
    }

    LOGGER_INFO("  test_unified_allocate");

    const device_unique_id_t gpu = 1; // first non-host device
    const size_t size = 1 << 20;

    unsigned char * mem = (unsigned char *) runtime.memory_unified_allocate(gpu, size);
    assert(mem != NULL);

    // unified memory is accessible from the host
    for (size_t i = 0 ; i < size ; ++i)
        mem[i] = (unsigned char) (i % 256);
    for (size_t i = 0 ; i < size ; ++i)
        assert(mem[i] == (unsigned char) (i % 256));

    runtime.memory_unified_deallocate(gpu, mem, size);
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    test_host_allocate(runtime);
    test_unified_allocate(runtime);

    assert(runtime.deinit() == 0);

    return 0;
}
