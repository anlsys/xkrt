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

# include <xkrt/memory/access/blas/dependency-tree.hpp>
# include <xkrt/runtime.h>
# include <xkrt/task/format.h>
# include <xkrt/task/task.hpp>

# include <assert.h>
# include <string.h>

XKRT_NAMESPACE_USE;

static int r = 0;

static void
func(task_t * task)
{
    r = 1;
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    // create an empty task format
    task_format_id_t fmtid;
    {
        task_format_t format;
        memset(&format, 0, sizeof(task_format_t));
        format.f[XKRT_TASK_FORMAT_TARGET_HOST] = (task_format_func_t) func;
        fmtid = runtime.task_format_create(&format);
    }
    assert(fmtid);

    thread_t * thread = thread_t::get_tls();
    assert(thread);

    // Create a task
    # define AC 1
    constexpr task_flag_bitfield_t flags = TASK_FLAG_ACCESSES;
    constexpr size_t task_size = task_compute_size(flags, AC);
    constexpr void * args = NULL;
    constexpr size_t args_size = 0;

    task_t * task = runtime.task_new(fmtid, flags, args, args_size, AC);

    task_acs_info_t * acs = TASK_ACS_INFO(task);
    new (acs) task_acs_info_t(AC);

    # if XKRT_SUPPORT_DEBUG
    snprintf(task->label, sizeof(task->label), "dependent-task-test");
    # endif

    // set accesses
    access_t * accesses = TASK_ACCESSES(task);
    {
        static_assert(AC <= XKRT_TASK_MAX_ACCESSES);

        const int ld = 1;
        int * mem = (int *) malloc(ld * ld * sizeof(int));
        const int  m = ld;
        const int  n = ld;
        for (int i = 0 ; i < m ; ++i)
            for (int j = 0 ; j < m ; ++j)
                mem[i*ld+j] = 42;

        new (accesses + 0) access_t(task, MATRIX_COLMAJOR, mem, ld, 0, 0, m, n, sizeof(int), ACCESS_MODE_R);

        runtime.task_accesses_resolve(accesses, AC);

        # undef AC
    }

    // submit it to the runtime
    runtime.task_commit(task);

    // wait
    runtime.task_wait();

    // deinit has an implicit taskwait
    assert(runtime.deinit() == 0);
    assert(r == 1);

    return 0;
}
