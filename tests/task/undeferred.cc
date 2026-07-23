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

/*
 * XKRT-native undeferred-task (OpenMP `if(0)`) test.
 *
 * An undeferred task follows the NORMAL scheduling path (any worker of the team
 * may execute it); the only extra semantic is that the thread that spawned it
 * suspends until THAT specific task completed -- runtime_t::task_wait(task_t*).
 * We verify the spawning thread does not proceed until the task has run.
 */

# include <xkrt/runtime.h>
# include <xkrt/task/task.hpp>

# include <assert.h>
# include <string.h>

XKRT_NAMESPACE_USE;

# define NB 64

static runtime_t runtime;
static int done[NB];

static void *
main_team(runtime_t * rt, team_t * team, thread_t * thread)
{
    if (thread->tid == 0)
    {
        for (int i = 0; i < NB; ++i)
        {
            // spawn a task on the normal path (any worker may run it) ...
            task_t * task = runtime.task_instanciate<TASK_FLAG_ZERO>(
                XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID, 0, nullptr, nullptr,
                [i] (runtime_t *, device_t *, task_t *) { done[i] = 1; }
            );
            task->flags |= TASK_FLAG_UNDEFERABLE;
            runtime.task_commit(task);

            // ... then suspend until this specific task completed (undeferred)
            runtime.task_wait(task);

            // by here, the undeferred task has necessarily run
            assert(done[i] == 1);
        }
    }
    runtime.team_barrier<true>(team, thread);
    return NULL;
}

int
main(void)
{
    assert(runtime.init() == 0);
    memset(done, 0, sizeof(done));

    team_t team;
    team.desc.nthreads = 4;
    team.desc.routine = (team_routine_t) main_team;
    team.desc.master_is_member = false;

    runtime.team_create(&team);
    runtime.team_join(&team);

    assert(runtime.deinit() == 0);

    return 0;
}
