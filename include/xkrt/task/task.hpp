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

#ifndef __XKRT_TASK_HPP__
# define __XKRT_TASK_HPP__

// https://stackoverflow.com/questions/45342776/how-to-include-c11-headers-when-compiling-c-with-gcc
// cannot use <stdatomic.h> with c++
// # include <stdatomic.h>
# include <atomic>
# include <functional>

# include <assert.h>
# include <stdint.h>
# include <stdio.h>

# include <xkrt/consts.h>
# include <xkrt/command/command.hpp>
# include <xkrt/data-structures/memory-pool.h>
# include <xkrt/data-structures/small-vector.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/todo.h>
# include <xkrt/memory/access/access.hpp>
# include <xkrt/memory/access/coherency-controller.hpp>
# include <xkrt/memory/access/dependency-domain.hpp>
# include <xkrt/memory/cache-line-size.hpp>
# include <xkrt/sync/spinlock.h>
# include <xkrt/task/flag.h>
# include <xkrt/task/format.h>
# include <xkrt/task/state.h>
# include <xkrt/types.h>

XKRT_NAMESPACE_BEGIN

# if XKRT_SUPPORT_DEBUG
#  define LOGGER_DEBUG_TASK_STATE(task)                                                                     \
    do {                                                                                                    \
        LOGGER_DEBUG("task `%s` of addr `%p` is now in state `%s`", task->label, task, xkrt_task_state_to_str(task->state.value));  \
    } while (0)
# else
#  define LOGGER_DEBUG_TASK_STATE(task)
# endif

typedef struct  task_t
{
    public:

        /* parent task */
        task_t * parent;

        /* children counter - number of uncompleted children tasks */
        std::atomic<uint32_t> cc;

        /* task state */
        struct {
            spinlock_t      lock;
            task_state_t    value;
        } state;

        /* task format id */
        task_format_id_t fmtid;

        /* task flags */
        task_flag_bitfield_t flags;

        # if XKRT_SUPPORT_DEBUG
        char label[128];
        # endif /* XKRT_SUPPORT_DEBUG */

    public:

        task_t(task_format_id_t fmtid, task_flag_bitfield_t flags) :
            parent(NULL),
            cc(0),
            state { .lock = SPINLOCK_INITIALIZER, .value = TASK_STATE_ALLOCATED },
            fmtid(fmtid),
            flags(flags)
        {
            # if XKRT_SUPPORT_DEBUG
            strncpy(this->label, "(unamed)", sizeof(this->label));
            # endif
        }

}               task_t;

/* task dependencies infos */
typedef struct  task_acs_info_t
{
    /*
     * wait counter
     * - if dependent task, it may be scheduled once it reached 0
     * - if detachable task, it is completed when it reached 2
     */
    task_wait_counter_t wc;

    /* access counter (number of accesses) */
    task_access_counter_t ac;

    /* constructor, wc is initially '1' as task must be commited */
    task_acs_info_t(task_access_counter_t ac) : wc(1), ac(ac) {}

}               task_acs_info_t;

/* detachable counter, shared with 'task_acs_info_t' if the task is both DEPENDENT and DETACHABLE */
typedef struct  task_det_info_t
{
    task_wait_counter_t wc;
    task_det_info_t()                               : task_det_info_t(0) {}
    task_det_info_t(task_wait_counter_type_t value) : wc(value)          {}
}               task_det_info_t;

typedef struct  task_dev_info_t
{
    /* worker id on where to schedule once ready (or 'XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID' if leaving the decision to the scheduler) */
    device_unique_id_t targeted_device_id;

    // TODO : move the 'ocr' field to the 'dep_info' : it is tied to accesses, not to a device

    /* execute on the device that owns a copy of the access at accesses[ocr_access_index]
     * If 'XKRT_UNSPECIFIED_TASK_ACCESS', leave the decision to the scheduler */
    task_access_counter_t ocr_access_index;

    /* the elected device on which that task got scheduled */
    device_unique_id_t elected_device_unique_id;

    /* constructor */
    task_dev_info_t(device_unique_id_t target, task_access_counter_t ocr)
        : targeted_device_id(target), ocr_access_index(ocr), elected_device_unique_id(XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID) {}

}               task_dev_info_t;

/* info about domain of dependencies */
typedef struct  task_dom_info_t
{
    /* dependency controller - only the thread currently executing the task may read this list */
    struct {
        DependencyDomain * handle;
        DependencyDomain * interval;
        small_vector_t<DependencyDomain *, 4> blas;
    } deps;

    /* memory controller for coherency - all threads may try to access this list */
    struct {
        MemoryCoherencyController * handle;

        struct {
            MemoryCoherencyController * mcc;
            spinlock_t lock;
        } interval;

        struct {
            small_vector_t<MemoryCoherencyController *, 4> mcc;
            spinlock_t lock;
        } blas;

    } mccs;

    task_dom_info_t() : deps{}, mccs{} {}

}               task_dom_info_t;

/* moldability info */
typedef struct  task_mol_info_t
{
    /* return true if that task should be split, false otherwise */
    const std::function<bool(task_t *, access_t *)> & split_condition;

    /* the task args size */
    const size_t args_size;

    task_mol_info_t(
        const std::function<bool(task_t *, access_t *)> & split_condition,
        const size_t args_size = 0
    ) :
        split_condition(split_condition),
        args_size(args_size)
    {}

    ~task_mol_info_t() {}

}               task_mol_info_t;

/* a task graph */
class task_dependency_graph_t
{
    public:

        /* list of root tasks */
        std::vector<task_t *> roots;

        /* list of leaf tasks */
        std::vector<task_t *> leaves;

        /* list of all tasks */
        std::vector<task_t *> tasks;

        /* spinlock to push to the roots list */
        spinlock_t roots_lock;

        /* visited flag to use when visiting tasks */
        bool visited_flag_cmp;

    public:
        /* constructor/destructor */
        task_dependency_graph_t(void) : roots(), roots_lock(0), visited_flag_cmp(true) {}
        ~task_dependency_graph_t(void) {}

        /* methods helper */
        void foreach_task(std::function<void(task_t *)> f);
        void dump_tasks(FILE * f);
        void dump_accesses(FILE * f);

        /* optimize the graph after recording it */
        void postprocess(void);

        /* return the number of tasks in the tdg */
        size_t get_ntasks(void)
        {
            return this->tasks.size();
        }

    private:
        void walk(std::function<void(task_t *)> f);

        void list_tasks(void);
        void remove_transitive_edges(void);
        void compute_leaves(void);
};

/* graph info */
typedef struct  task_gph_info_t
{
    /* graph currently being recorded */
    task_dependency_graph_t * tdg;

    /* constructor / destructor */
    task_gph_info_t(
        task_dependency_graph_t * tdg
    ) :
        tdg(tdg)
    {}

    ~task_gph_info_t(void) {}

}               task_gph_info_t;

/* a recorded command */
struct task_command_record_t
{
    /* the command */
    command_t command;

    /* the state when the command was emitted */
    task_state_t state;
};

/* record info */
typedef struct  task_rec_info_t
{
    /* list of commands recorded */
    memory_pool_t<task_command_record_t, XKRT_TASK_RECORD_COMMAND_POOL_CHUNK_CAPACITY> commands;

    /* list of predecessor accesses */
    small_vector_t<access_t *, 8> predecessors;

    /* list of successor accesses */
    small_vector_t<access_t *, 8> successors;

    /* flag whether this task had been visited (0 or 1) */
    bool visited_flag;

    /* index of the task in the taskgraph */
    size_t index;

    /* constructor/destructor */
    task_rec_info_t(void) : commands(), predecessors(), successors(), visited_flag(false), index(SIZE_MAX) {}
    ~task_rec_info_t(void) {}

}               task_rec_info_t;

/* Push a new command in the command records of the passed task */
command_t * task_put_command_record(task_t * task);

/* fallback if wrong flags parameter -
 * https://stackoverflow.com/questions/20461121/constexpr-error-at-compile-time-but-no-overhead-at-run-time */
static size_t
task_get_base_size_fallback(task_flag_bitfield_t flags)
{
    LOGGER_FATAL("Invalid task flag combination: `%u`", flags);
    return 0;
}

/**
 *  In case of a task with all flags, its memory is
 *   _________________________________________________________________________________________________________________________________________________________
 *  |                                                                                                                                                         |
 *  | task_t | task_acs_info_t | task_det_info | task_dev_info_t | task_dom_info_t | task_mol_info_t | task_gph_info_t | task_rec_info_t | access_t... | args |
 *  |_________________________________________________________________________________________________________________________________________________________|
 *
 * if some flags are removed, builing blocks are removed
 */

////////////////////////////////////
// (start) CODE HERE IS GENERATED //
////////////////////////////////////

static constexpr size_t
task_get_extra_size(const task_flag_bitfield_t flags)
{
    switch (flags)
    {
        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +                          0 +                          0 +                          0 +                          0); // 0.0.0.0.0.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.0.0.0.0.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 0.0.0.0.0.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.0.0.0.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.0.0.0.1.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.0.0.0.1.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.0.0.0.1.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.0.0.1.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 0.0.0.1.0.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.0.0.1.0.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 0.0.0.1.0.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.0.1.0.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.0.0.1.1.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.0.0.1.1.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.0.0.1.1.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.0.1.1.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +                          0); // 0.0.1.0.0.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.0.1.0.0.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 0.0.1.0.0.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.1.0.0.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.0.1.0.1.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.0.1.0.1.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.0.1.0.1.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.1.0.1.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 0.0.1.1.0.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.0.1.1.0.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 0.0.1.1.0.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.1.1.0.1.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.0.1.1.1.0.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.0.1.1.1.0.1

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.0.1.1.1.1.0

        case (                  TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.0.1.1.1.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +                          0 +                          0); // 0.1.0.0.0.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.1.0.0.0.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 0.1.0.0.0.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.0.0.0.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.1.0.0.1.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.1.0.0.1.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.1.0.0.1.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.0.0.1.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 0.1.0.1.0.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.1.0.1.0.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 0.1.0.1.0.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.0.1.0.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.1.0.1.1.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.1.0.1.1.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.1.0.1.1.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.0.1.1.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +                          0); // 0.1.1.0.0.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.1.1.0.0.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 0.1.1.0.0.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.1.0.0.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.1.1.0.1.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.1.1.0.1.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.1.1.0.1.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.1.0.1.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 0.1.1.1.0.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 0.1.1.1.0.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 0.1.1.1.0.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.1.1.0.1.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 0.1.1.1.1.0.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 0.1.1.1.1.0.1

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 0.1.1.1.1.1.0

        case (                  TASK_FLAG_ZERO |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (                         0 +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 0.1.1.1.1.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +                          0 +                          0 +                          0); // 1.0.0.0.0.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.0.0.0.0.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 1.0.0.0.0.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.0.0.0.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.0.0.0.1.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.0.0.0.1.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.0.0.0.1.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.0.0.1.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 1.0.0.1.0.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.0.0.1.0.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 1.0.0.1.0.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.0.1.0.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.0.0.1.1.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.0.0.1.1.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.0.0.1.1.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.0.1.1.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +                          0); // 1.0.1.0.0.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.0.1.0.0.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 1.0.1.0.0.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.1.0.0.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.0.1.0.1.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.0.1.0.1.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.0.1.0.1.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.1.0.1.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 1.0.1.1.0.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.0.1.1.0.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 1.0.1.1.0.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.1.1.0.1.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.0.1.1.1.0.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.0.1.1.1.0.1

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.0.1.1.1.1.0

        case (                TASK_FLAG_RECORD |             TASK_FLAG_ZERO |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +                          0 +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.0.1.1.1.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +                          0 +                          0); // 1.1.0.0.0.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.1.0.0.0.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 1.1.0.0.0.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.0.0.0.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.1.0.0.1.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.1.0.0.1.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.1.0.0.1.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.0.0.1.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 1.1.0.1.0.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.1.0.1.0.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 1.1.0.1.0.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.0.1.0.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.1.0.1.1.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.1.0.1.1.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.1.0.1.1.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |             TASK_FLAG_ZERO |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +                          0 +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.0.1.1.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +                          0); // 1.1.1.0.0.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.1.1.0.0.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +                          0); // 1.1.1.0.0.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.1.0.0.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.1.1.0.1.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.1.1.0.1.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.1.1.0.1.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |             TASK_FLAG_ZERO |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +                          0 +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.1.0.1.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +                          0); // 1.1.1.1.0.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +                          0 +    sizeof(task_acs_info_t)); // 1.1.1.1.0.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +                          0); // 1.1.1.1.0.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |             TASK_FLAG_ZERO |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +                          0 +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.1.1.0.1.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +                          0); // 1.1.1.1.1.0.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |             TASK_FLAG_ZERO |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +                          0 +    sizeof(task_acs_info_t)); // 1.1.1.1.1.0.1

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |             TASK_FLAG_ZERO):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +                          0); // 1.1.1.1.1.1.0

        case (                TASK_FLAG_RECORD |            TASK_FLAG_GRAPH |         TASK_FLAG_MOLDABLE |           TASK_FLAG_DOMAIN |           TASK_FLAG_DEVICE |       TASK_FLAG_DETACHABLE |         TASK_FLAG_ACCESSES):
            return (   sizeof(task_rec_info_t) +    sizeof(task_gph_info_t) +    sizeof(task_mol_info_t) +    sizeof(task_dom_info_t) +    sizeof(task_dev_info_t) +    sizeof(task_det_info_t) +    sizeof(task_acs_info_t)); // 1.1.1.1.1.1.1

        default:
            return task_get_base_size_fallback(flags);
    }
}

static constexpr task_acs_info_t *
TASK_ACS_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_ACCESSES);
    return (task_acs_info_t *) (task + 1);
}

static task_acs_info_t *
TASK_ACS_INFO(const task_t * task)
{
    return TASK_ACS_INFO(task, task->flags);
}

static constexpr task_det_info_t *
TASK_DET_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_DETACHABLE);
    if (flags & TASK_FLAG_ACCESSES)
        return (task_det_info_t *) (TASK_ACS_INFO(task, flags) + 1);
    return (task_det_info_t *) (task + 1);
}

static task_det_info_t *
TASK_DET_INFO(const task_t * task)
{
    return TASK_DET_INFO(task, task->flags);
}

static constexpr task_dev_info_t *
TASK_DEV_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_DEVICE);
    if (flags & TASK_FLAG_DETACHABLE)
        return (task_dev_info_t *) (TASK_DET_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_ACCESSES)
        return (task_dev_info_t *) (TASK_ACS_INFO(task, flags) + 1);
    return (task_dev_info_t *) (task + 1);
}

static task_dev_info_t *
TASK_DEV_INFO(const task_t * task)
{
    return TASK_DEV_INFO(task, task->flags);
}

static constexpr task_dom_info_t *
TASK_DOM_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_DOMAIN);
    if (flags & TASK_FLAG_DEVICE)
        return (task_dom_info_t *) (TASK_DEV_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DETACHABLE)
        return (task_dom_info_t *) (TASK_DET_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_ACCESSES)
        return (task_dom_info_t *) (TASK_ACS_INFO(task, flags) + 1);
    return (task_dom_info_t *) (task + 1);
}

static task_dom_info_t *
TASK_DOM_INFO(const task_t * task)
{
    return TASK_DOM_INFO(task, task->flags);
}

static constexpr task_mol_info_t *
TASK_MOL_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_MOLDABLE);
    if (flags & TASK_FLAG_DOMAIN)
        return (task_mol_info_t *) (TASK_DOM_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DEVICE)
        return (task_mol_info_t *) (TASK_DEV_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DETACHABLE)
        return (task_mol_info_t *) (TASK_DET_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_ACCESSES)
        return (task_mol_info_t *) (TASK_ACS_INFO(task, flags) + 1);
    return (task_mol_info_t *) (task + 1);
}

static task_mol_info_t *
TASK_MOL_INFO(const task_t * task)
{
    return TASK_MOL_INFO(task, task->flags);
}

static constexpr task_gph_info_t *
TASK_GPH_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_GRAPH);
    if (flags & TASK_FLAG_MOLDABLE)
        return (task_gph_info_t *) (TASK_MOL_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DOMAIN)
        return (task_gph_info_t *) (TASK_DOM_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DEVICE)
        return (task_gph_info_t *) (TASK_DEV_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DETACHABLE)
        return (task_gph_info_t *) (TASK_DET_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_ACCESSES)
        return (task_gph_info_t *) (TASK_ACS_INFO(task, flags) + 1);
    return (task_gph_info_t *) (task + 1);
}

static task_gph_info_t *
TASK_GPH_INFO(const task_t * task)
{
    return TASK_GPH_INFO(task, task->flags);
}

static constexpr task_rec_info_t *
TASK_REC_INFO(const task_t * task, const task_flag_bitfield_t flags)
{
    assert(flags & TASK_FLAG_RECORD);
    if (flags & TASK_FLAG_GRAPH)
        return (task_rec_info_t *) (TASK_GPH_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_MOLDABLE)
        return (task_rec_info_t *) (TASK_MOL_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DOMAIN)
        return (task_rec_info_t *) (TASK_DOM_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DEVICE)
        return (task_rec_info_t *) (TASK_DEV_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_DETACHABLE)
        return (task_rec_info_t *) (TASK_DET_INFO(task, flags) + 1);
    if (flags & TASK_FLAG_ACCESSES)
        return (task_rec_info_t *) (TASK_ACS_INFO(task, flags) + 1);
    return (task_rec_info_t *) (task + 1);
}

static task_rec_info_t *
TASK_REC_INFO(const task_t * task)
{
    return TASK_REC_INFO(task, task->flags);
}

//////////////////////////////////
// (end) CODE HERE IS GENERATED //
//////////////////////////////////


/* Given flags and the number of accesses, computes (at compile-time) the size
 * in bytes required for the task (without args_t) */
static constexpr inline size_t
task_compute_size(const task_flag_bitfield_t flags, const task_access_counter_t ac)
{
    return sizeof(task_t) + task_get_extra_size(flags) + ac*sizeof(access_t);
}

static constexpr size_t
TASK_ACCESSES_OFFSET(const task_flag_bitfield_t flags)
{
    // accesses must be stored right after the task struct
    return sizeof(task_t) + task_get_extra_size(flags);
}

static inline access_t *
TASK_ACCESSES(const task_t * task, const task_flag_bitfield_t flags)
{
    return (access_t *) (((char *) task) + TASK_ACCESSES_OFFSET(flags));
}

static inline access_t *
TASK_ACCESSES(const task_t * task)
{
    return TASK_ACCESSES(task, task->flags);
}

static inline void *
TASK_ARGS(const task_t * task, const size_t task_size)
{
    return (void *) (((char *) task) + task_size);
}

static inline size_t
TASK_SIZE(const task_t * task)
{
    if (task->flags & TASK_FLAG_ACCESSES)
    {
        task_acs_info_t * acs = TASK_ACS_INFO(task);
        return sizeof(task_t) + task_get_extra_size(task->flags) + acs->ac*sizeof(access_t);
    }
    else
        return sizeof(task_t) + task_get_extra_size(task->flags);
}

static inline void *
TASK_ARGS(task_t * task)
{
    return TASK_ARGS(task, TASK_SIZE(task));
}

///////////////////////////////////
// Methods to setup dependencies //
///////////////////////////////////

# define XKRT_TASK_DEPENDENCE_ALREADY_SET   0
# define XKRT_TASK_DEPENDENCE_SET           1
# define XKRT_TASK_DEPENDENCE_SKIPPED       2

/* pred precedes succ - call 'F(args)' if 'pred' isnt completed yet in a lock region */
template <typename... Args>
static inline int
__task_precedes(
    task_t * pred,
    task_t * succ,
    void (*F)(Args...),
    Args... args
) {
    assert(pred);
    assert(succ);
    assert(pred != succ);   // this failing most likely means you have 2 accesses overlaping
    assert(pred->state.value >= TASK_STATE_ALLOCATED);
    assert(succ->state.value >= TASK_STATE_ALLOCATED);
    assert(pred->flags & TASK_FLAG_ACCESSES);
    assert(succ->flags & TASK_FLAG_ACCESSES);

    int r = XKRT_TASK_DEPENDENCE_SKIPPED;
    if ((volatile task_state_t) pred->state.value < TASK_STATE_COMPLETED)
    {
        SPINLOCK_LOCK(pred->state.lock);
        {
            if ((volatile task_state_t) pred->state.value < TASK_STATE_COMPLETED)
            {
                LOGGER_DEBUG("Dependency: `%s` -> `%s`", pred->label, succ->label);
                task_acs_info_t * sacs = TASK_ACS_INFO(succ);
                sacs->wc.fetch_add(1, std::memory_order_seq_cst);
                F(std::forward<Args>(args)...);
                r = XKRT_TASK_DEPENDENCE_SET;
            }
        }
        SPINLOCK_UNLOCK(pred->state.lock);
    }
    return r;
}

static inline void
__access_link(access_t * pred, access_t * succ)
{
    pred->successors.push_back(succ);
}

inline int
__access_precedes(access_t * pred, access_t * succ)
{
    // succ must be a dependent task
    assert(succ->task->flags & TASK_FLAG_ACCESSES);

    // succ must have a wc>0 at this point: we are still processing dependencies, it cannot be scheduled yet
    assert(TASK_ACS_INFO(succ->task)->wc > 0);

    // succ has reached the maximum number of dependencies
    assert(TASK_ACS_INFO(succ->task)->wc < ((1 << (8 * sizeof(task_wait_counter_type_t))) - 1));

    // avoid redundant edges
    if (pred->successors.size() && pred->successors.back()->task == succ->task)
        return XKRT_TASK_DEPENDENCE_ALREADY_SET;

    // TODO: what if only one is being recorded? (i.e., omp taskgraph nogroup)
    /* if both tasks are being recorded */
    if ((pred->task->flags & TASK_FLAG_RECORD) && (succ->task->flags & TASK_FLAG_RECORD))
    {
        task_rec_info_t * pred_rec = TASK_REC_INFO(pred->task);
        task_rec_info_t * succ_rec = TASK_REC_INFO(succ->task);
        assert(pred_rec->successors.back() != succ);
        pred_rec->successors.push_back(succ);
        succ_rec->predecessors.push_back(pred);
    }

    // set edge
    return __task_precedes(pred->task, succ->task, __access_link, pred, succ);
}

////////////////////////////////////
// Methods to transition the task //
////////////////////////////////////

/* mark the task ready and call F(args) */
template <typename... Args>
static inline void
__task_ready(
    task_t * task,
    void (*F)(Args..., task_t *),
    Args... args
) {
    assert(task->state.value == TASK_STATE_ALLOCATED);
    assert(!(task->flags & TASK_FLAG_ACCESSES) || (TASK_ACS_INFO(task)->wc.load() == 0));
    task->state.value = TASK_STATE_READY;
    LOGGER_DEBUG_TASK_STATE(task);
    if (F)
        F(std::forward<Args>(args)..., task);

    /* the task was marked ready, so it cannot have any more predecessors now.
     * If is being recorded and has no predecessor, it is a root of its taskgraph. */
    if (task->flags & TASK_FLAG_RECORD)
    {
        task_rec_info_t * rec = TASK_REC_INFO(task);
        assert(rec);

        if (rec->predecessors.size() == 0)
        {
            assert(task->parent);
            assert(task->parent->flags & TASK_FLAG_GRAPH);

            task_gph_info_t * gph = TASK_GPH_INFO(task->parent);
            assert(gph);
            assert(gph->tdg);

            SPINLOCK_LOCK(gph->tdg->roots_lock);
            {
                gph->tdg->roots.push_back(task);
            }
            SPINLOCK_UNLOCK(gph->tdg->roots_lock);
        }
    }
}

/* commit the task and call F(args) if it is now ready */
template <typename... Args>
static inline void
__task_commit(
    task_t * task,
    void (*F)(Args..., task_t *),
    Args... args
) {
    assert(task->state.value == TASK_STATE_ALLOCATED);
    if (task->flags & TASK_FLAG_ACCESSES)
    {
        task_acs_info_t * acs = TASK_ACS_INFO(task);
        if (acs->wc.fetch_sub(1, std::memory_order_seq_cst) == 1)
            __task_ready(task, F, args...);
    }
    else
        __task_ready(task, F, args...);
}

static inline void
__task_fetching(
    task_wait_counter_type_t n,
    task_t * task
) {
    assert(task->flags & TASK_FLAG_ACCESSES);
    task_acs_info_t * acs = TASK_ACS_INFO(task);
    if (acs->wc.fetch_add(n, std::memory_order_seq_cst) == 0)
    {
        assert(task->state.value == TASK_STATE_READY ||
                task->state.value == TASK_STATE_ALLOCATED);
        task->state.value = TASK_STATE_DATA_FETCHING;
        LOGGER_DEBUG_TASK_STATE(task);
    }
}

/* notify that 'n' accesses had been fetched. If all accesses were fetched,
 * then mark the task as 'fetched' and call F(...) */
template <typename... Args>
static inline void
__task_fetched(
    task_wait_counter_type_t n,
    task_t * task,
    void (*F)(Args..., task_t *),
    Args... args
) {
    // allocated means it was fetched as part of prefetching
    assert(task->state.value == TASK_STATE_ALLOCATED ||
            task->state.value == TASK_STATE_DATA_FETCHING);
    assert(task->flags & TASK_FLAG_ACCESSES);
    task_acs_info_t * acs = TASK_ACS_INFO(task);
    if (acs->wc.fetch_sub(n, std::memory_order_seq_cst) == n)
    {
        task->state.value = TASK_STATE_DATA_FETCHED;
        LOGGER_DEBUG_TASK_STATE(task);
        F(std::forward<Args>(args)..., task);
    }
}

inline device_unique_id_t
task_get_device_unique_id(task_t * task)
{
    if (task->flags & TASK_FLAG_DEVICE)
    {
        task_dev_info_t * dev = TASK_DEV_INFO(task);
        return dev->elected_device_unique_id;
    }
    return XKRT_HOST_DEVICE_UNIQUE_ID;
}

XKRT_NAMESPACE_END

# endif /* __XKRT_TASK_HPP__ */
