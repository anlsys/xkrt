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
 *  team_critical_begin / team_critical_end: mutual exclusion across team
 *  threads. Every thread performs ITERS unprotected read-modify-writes of a
 *  shared NON-atomic counter inside the critical section. A broken critical
 *  section would lose updates and the final count would be < expected.
 */

# ifndef _GNU_SOURCE
#  define _GNU_SOURCE
# endif
# include <sched.h>

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>

# include <assert.h>

XKRT_NAMESPACE_USE;

static const int ITERS = 200;

/* deliberately NON-atomic: correctness relies on the critical section */
static long counter = 0;

static void *
main_team(runtime_t * runtime, team_t * team, thread_t * thread)
{
    (void) thread;
    for (int i = 0 ; i < ITERS ; ++i)
    {
        runtime->team_critical_begin(team);
        long v = counter;
        sched_yield();          // widen the race window
        counter = v + 1;
        runtime->team_critical_end(team);
    }
    return NULL;
}

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    team_t team;
    team.desc.nthreads = 0;     // all cpus
    team.desc.routine = (team_routine_t) main_team;
    team.desc.master_is_member = true;

    counter = 0;
    runtime.team_create(&team);
    runtime.team_join(&team);

    const long expected = (long) team.priv.nthreads * ITERS;
    LOGGER_INFO("counter = %ld, expected = %ld (%d threads x %d iters)",
            counter, expected, team.priv.nthreads, ITERS);
    assert(counter == expected);

    assert(runtime.deinit() == 0);

    return 0;
}
