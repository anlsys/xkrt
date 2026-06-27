/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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

/* C API: device count query + synchronous/asynchronous memory registration. */

# include <assert.h>
# include <stdlib.h>
# include <xkrt/xkrt.h>

int
main(void)
{
    xkrt_runtime_t * runtime;
    assert(xkrt_init(&runtime) == 0);

    /* at least the host device is always present */
    unsigned int ndevices = xkrt_get_ndevices(runtime);
    assert(ndevices >= 1);

    const size_t size = 64 * 1024;
    void * ptr = malloc(size);
    assert(ptr);

    /* synchronous register / unregister */
    assert(xkrt_memory_register(runtime, ptr, size) == 0);
    assert(xkrt_memory_unregister(runtime, ptr, size) == 0);

    /* asynchronous register / unregister */
    assert(xkrt_memory_register_async(runtime, ptr, size) == 0);
    xkrt_task_wait(runtime);
    assert(xkrt_memory_unregister_async(runtime, ptr, size) == 0);
    xkrt_task_wait(runtime);

    free(ptr);

    assert(xkrt_deinit(runtime) == 0);

    return 0;
}
