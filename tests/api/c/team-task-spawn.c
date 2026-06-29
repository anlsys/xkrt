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

/* C API: retrieve a driver team and spawn several tasks onto it. */

# include <assert.h>
# include <xkrt/xkrt.h>

# define N 8

static void
task_func(
    xkrt_runtime_t * runtime,
    xkrt_device_t * device,
    xkrt_task_t * task,
    void * user_data
) {
    (void) runtime;
    (void) device;
    (void) task;
    /* each task writes its own slot: no data race */
    *((int *) user_data) = 1;
}

int
main(void)
{
    xkrt_runtime_t * runtime;
    assert(xkrt_init(&runtime) == 0);

    xkrt_team_t * team = xkrt_team_driver_get(runtime, XKRT_DRIVER_TYPE_HOST);
    assert(team);

    int done[N];
    for (int i = 0 ; i < N ; ++i)
        done[i] = 0;

    for (int i = 0 ; i < N ; ++i)
        xkrt_team_task_spawn(runtime, team, task_func, &done[i]);

    xkrt_task_wait(runtime);

    for (int i = 0 ; i < N ; ++i)
        assert(done[i] == 1);

    assert(xkrt_deinit(runtime) == 0);

    return 0;
}
