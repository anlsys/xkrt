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
 *  Exercises the *incremental* task-format API:
 *      task_format_put(label)             -> reserve an id (empty format)
 *      task_format_set(id, target, func)  -> attach an implementation
 *      task_format_get(id)                -> retrieve the format
 *
 *  (as opposed to task_format_create() which registers an already-filled
 *  format in one call, already covered by task-format.cc)
 */

# include <xkrt/runtime.h>
# include <xkrt/task/format.h>
# include <xkrt/task/task.hpp>

# include <assert.h>
# include <string.h>

XKRT_NAMESPACE_USE;

static int ran = 0;

/* host task formats are always called as (runtime, device, task) */
static void
body(runtime_t * runtime, device_t * device, task_t * task)
{
    (void) runtime;
    (void) device;
    (void) task;
    ran = 1;
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    // reserve a fresh format id
    task_format_id_t fmtid = runtime.task_format_put("incremental");
    assert(fmtid != XKRT_TASK_FORMAT_NULL);

    // the format exists but has no implementation yet
    task_format_t * fmt = runtime.task_format_get(fmtid);
    assert(fmt != NULL);
    assert(fmt->f[XKRT_TASK_FORMAT_TARGET_HOST] == NULL);
    assert(strcmp(fmt->label, "incremental") == 0);

    // attach a host implementation
    int rc = runtime.task_format_set(fmtid, XKRT_TASK_FORMAT_TARGET_HOST, (task_format_func_t) body);
    assert(rc == 0);

    // the implementation is now visible
    fmt = runtime.task_format_get(fmtid);
    assert(fmt != NULL);
    assert(fmt->f[XKRT_TASK_FORMAT_TARGET_HOST] == (task_format_func_t) body);

    // two consecutive puts must yield distinct ids
    task_format_id_t other = runtime.task_format_put("incremental-2");
    assert(other != fmtid);

    // spawn a task using the incrementally-built format
    constexpr task_flag_bitfield_t flags = TASK_FLAG_ZERO;
    task_t * task = runtime.task_new(fmtid, flags, NULL, 0, 0);
    runtime.task_commit(task);
    runtime.task_wait();

    assert(runtime.deinit() == 0);
    assert(ran == 1);

    return 0;
}
