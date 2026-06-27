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

# include <xkrt/runtime.h>
# include <assert.h>

XKRT_NAMESPACE_USE;

static runtime_t runtime;
static volatile int run_for_device[XKRT_DEVICES_MAX];

static void *
run(runtime_t * rt, team_t * team, thread_t * thread)
{
    assert(thread->tid >= 0);
    assert(thread->tid < runtime.drivers.devices.n);
    run_for_device[thread->tid] = 1;
    return NULL;
}

int
main(void)
{
    assert(runtime.init() == 0);

    team_t team;
    team.desc.args             = NULL;
    team.desc.binding.flags    = XKRT_TEAM_BINDING_FLAG_NONE;
    team.desc.binding.mode     = XKRT_TEAM_BINDING_MODE_COMPACT;
    team.desc.binding.places   = XKRT_TEAM_BINDING_PLACES_DEVICE;
    team.desc.master_is_member = false;
    team.desc.nthreads         = runtime.drivers.devices.n;
    team.desc.routine          = (team_routine_t) run;

    runtime.team_create(&team);
    runtime.team_join(&team);

    for (int i = 0 ; i < runtime.drivers.devices.n ; ++i)
        assert(run_for_device[i] == 1);

    assert(runtime.deinit() == 0);

    return 0;
}
