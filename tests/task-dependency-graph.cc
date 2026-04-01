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

# include <new>

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>

# if NDEBUG
#  define assert(X) X
# endif

XKRT_NAMESPACE_USE;

int
main(void)
{
    runtime_t runtime;

    assert(runtime.init() == 0);

    constexpr size_t size = 1024;
    uintptr_t X = (uintptr_t) malloc(size);
    uintptr_t Y = (uintptr_t) malloc(size);

    const device_unique_id_t device_unique_id = runtime.get_ndevices() - 1;
    assert(device_unique_id != XKRT_HOST_DEVICE_UNIQUE_ID);

    uint64_t t0 = get_nanotime();

    //  # pragma omp taskgraph
    //  {

    /* record a tdg (implicit task_wait) */
    constexpr bool execute_commands = true;
    task_dependency_graph_t tdg;
    runtime.task_dependency_graph_record_start(&tdg, execute_commands);

    std::atomic<int> run(0);

    // spawn N tasks
    # define N 4
    for (int i = 0 ; i < N ; ++i)
    {

        //  # pragma omp task access(read: X[i*size/N:size/N]) access(readwrite: Y[i*size/N:size/N]) device(device_unique_id)
        //      {}

        // spawn a task with 1 access
        constexpr task_flag_bitfield_t flags = TASK_FLAG_ACCESSES | TASK_FLAG_DEVICE;
        constexpr task_access_counter_t ac = 2;
        runtime.task_spawn<flags>(

            // device to use
            (device_unique_id + i) % 2,
            // device_unique_id,

            // number of accesses
            ac,

            // set accesses
            [=] (task_t * task, access_t * accesses) {

                // start index and block size
                const size_t start   = i*size/N;
                const size_t subsize = size/N;

                uintptr_t Xa = (((const uintptr_t) X) + start);
                uintptr_t Xb = Xa + subsize;

                uintptr_t Ya = (((const uintptr_t) Y) + start);
                uintptr_t Yb = Ya + subsize;

                assert(X <= Xa && Xb <= X + size);
                assert(Y <= Ya && Yb <= Y + size);

                // y := x + y
                new (accesses + 0) access_t(task, Xa, Xb, ACCESS_MODE_R);
                new (accesses + 1) access_t(task, Ya, Yb, ACCESS_MODE_RW);
            },

            // split condition
            nullptr,

            // routine
            [&] (runtime_t * runtime, device_t * device, task_t * task) {
                access_t * accesses = TASK_ACCESSES(task);
                const access_t * access = accesses + 0;
                const uintptr_t a = access->region.interval.segment[0].a;
                const uintptr_t b = access->region.interval.segment[0].b;
                LOGGER_INFO("Running [%lu, %lu]", a, b);
                ++run;
            }
        );
    }

    /* stop recording (implicit task wait) */
    runtime.task_dependency_graph_record_stop();

    //  } // pragma omp taskgraph

    uint64_t t1 = get_nanotime();

    # if 0
    {
        FILE * f = fopen("tasks.dot",    "w");
        tdg.dump_tasks(f);
        fclose(f);
    }

    {
        FILE * f = fopen("accesses.dot", "w");
        tdg.dump_accesses(f);
        fclose(f);
    }
    # endif

    uint64_t t2 = get_nanotime();

    # if 1

    //  Extract device commands generated from the task sequence
    //  Idk about this, OpenMP has no definitions of 'commands' -- which makes it hard to imagine
    //
    //  # pragma omp taskgraph filter(gpu-commands)

    /* build a cg from a tdg */
    command_graph_t cg;
    runtime.command_graph_from_task_dependency_graph(&tdg, &cg);

    uint64_t t3 = get_nanotime();

    /* replay the cg */
    runtime.command_graph_replay(&cg);

    uint64_t t4 = get_nanotime();

    /* remove useless nodes */
    //  # pragma omp taskgraph optimize(reduce_nodes)
    cg.optimize(ocg::COMMAND_GRAPH_PASS_REDUCE_NODE);

    /* remove redundant edges */
    //  # pragma omp taskgraph optimize(reduce_edges)
    cg.optimize(ocg::COMMAND_GRAPH_PASS_REDUCE_EDGE);

    /* contract the cg */
    //  # pragma omp taskgraph optimize(batch)
    cg.optimize(ocg::COMMAND_GRAPH_PASS_BATCH);

    uint64_t t5 = get_nanotime();

    /* replay N times */
    # define N_REPLAY 10
    uint64_t t_replay[N_REPLAY];
    for (int i = 0 ; i < N_REPLAY ; ++i)
    {
        uint64_t t0 = get_nanotime();
        runtime.command_graph_replay(&cg);
        uint64_t tf = get_nanotime();
        t_replay[i] = (tf - t0);
    }

    /* destroy the cg */
    runtime.command_graph_destroy(&cg);

    /* destroy tdg */
    runtime.task_dependency_graph_destroy(&tdg);

    printf("STATS\n");
    printf("  initial graph record + exec: %.4lf us\n", (t1 - t0) / 1e3);
    printf("         conversion tdg to cg: %.4lf us\n", (t3 - t2) / 1e3);
    printf("      non optimized cg replay: %.4lf us\n", (t4 - t3) / 1e3);
    printf("             cg optimizations: %.4lf us\n", (t5 - t4) / 1e3);
    for (int i = 0 ; i < N_REPLAY ; ++i)
        printf("          optimized replay %2d: %.4lf us\n", i, t_replay[i] / 1e3);
    # endif

    assert(runtime.deinit() == 0);

    return 0;
}
