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
 *  Host-only coverage of the task-dependency-graph recording API.
 *
 *  The existing `task-dependency-graph.cc` requires GPUs (it schedules tasks
 *  on non-host devices and builds a device command-graph). This test records
 *  a chain of *host* tasks instead, verifying that:
 *      - recording captures every spawned task,
 *      - tasks still execute (execute_commands = true) in dependency order.
 */

# include <xkrt/runtime.h>
# include <xkrt/task/task.hpp>
# include <xkrt/logger/logger.h>

# include <assert.h>

XKRT_NAMESPACE_USE;

static int value = 0;

int
main(void)
{
    runtime_t runtime;
    assert(runtime.init() == 0);

    constexpr int N = 8;
    value = 0;

    // record a chain of host tasks (RAW/WAW on a single point)
    task_dependency_graph_t tdg;
    constexpr bool execute_commands = true;
    runtime.task_dependency_graph_record_start(&tdg, execute_commands);

    for (int i = 0 ; i < N ; ++i)
    {
        runtime.task_spawn<1>(
            [] (task_t * task, access_t * accesses) {
                new (accesses + 0) access_t(task, (const void *) &value, ACCESS_MODE_RW);
            },
            [i] (runtime_t *, device_t *, task_t *) {
                // strict serialization: each task observes the previous value
                assert(value == i);
                value = i + 1;
            }
        );
    }

    // implicit task_wait: the recorded host tasks execute here
    runtime.task_dependency_graph_record_stop();

    // tasks executed in dependency order
    assert(value == N);

    // the graph captured every spawned task
    assert(tdg.get_ntasks() == (size_t) N);

    // replay/destroy are no-ops for a host graph but must be callable
    runtime.task_dependency_graph_replay(&tdg);
    runtime.task_dependency_graph_destroy(&tdg);

    assert(runtime.deinit() == 0);

    return 0;
}
