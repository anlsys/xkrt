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

# ifndef __XKRT_TEAM_H__
#  define __XKRT_TEAM_H__

#  include <xkrt/consts.h>
#  include <xkrt/memory/alignas.h>
#  include <xkrt/support.h>
#  include <xkrt/sync/spinlock.h>
#  include <xkrt/task/task.hpp>
#  include <xkrt/thread/deque.hpp>
#  include <xkrt/thread/naive-queue.hpp>
#  include <xkrt/thread/team-thread-place.h>

#  include <pthread.h>
#  include <atomic>
#  include <random>

#  include <linux/futex.h>      /* Definition of FUTEX_* constants */
#  include <sys/syscall.h>      /* Definition of SYS_* constants */
#  include <unistd.h>

/* pointer to the current team */
# define XKRT_TEAM_CURRENT (thread_t::get_tls()->team)

XKRT_NAMESPACE_BEGIN

/////////////
// THREADS //
/////////////

/* thread routine */
typedef void * (*team_routine_t)(void * runtime, void * team, void * thread);

struct thread_t;

/* A node in the hierarchical barrier tree */
typedef struct  team_hierarchy_node_t
{
    /* Thread IDs in this group at this level (static array) */
    int tids[XKRT_TEAM_HIERARCHY_GROUP_SIZE];
    int ntids;

}               team_hierarchy_node_t;

/* Hierarchical group structure for the entire team */
typedef struct  team_hierarchy_t
{
    /* Number of levels in the hierarchy */
    int nlevels;

    /* All nodes in the hierarchy */
    team_hierarchy_node_t * nodes;
    int nnodes;
}               team_hierarchy_t;

// THREAD BINDINGS

typedef enum    team_binding_mode_t
{
    XKRT_TEAM_BINDING_MODE_COMPACT,
    XKRT_TEAM_BINDING_MODE_SPREAD,
}               team_binding_mode_t;

typedef enum    team_binding_places_t
{
    XKRT_TEAM_BINDING_PLACES_HYPERTHREAD,
    XKRT_TEAM_BINDING_PLACES_CORE,
    XKRT_TEAM_BINDING_PLACES_L1,
    XKRT_TEAM_BINDING_PLACES_L2,
    XKRT_TEAM_BINDING_PLACES_L3,
    XKRT_TEAM_BINDING_PLACES_NUMA,
    XKRT_TEAM_BINDING_PLACES_DEVICE,
    XKRT_TEAM_BINDING_PLACES_SOCKET,
    XKRT_TEAM_BINDING_PLACES_MACHINE,
    XKRT_TEAM_BINDING_PLACES_EXPLICIT,
}               team_binding_places_t;

typedef enum    team_binding_flag_t
{
    XKRT_TEAM_BINDING_FLAG_NONE         = 0,
    XKRT_TEAM_BINDING_FLAG_EXCLUDE_HOST = (1 << 0)
}               team_binding_flag_t;

/* a place */
typedef cpu_set_t team_thread_place_t;

/**
 *  The supported combinations are:
 *    (mode = COMPACT, places = DEVICE)
 *      -> that will compactly bind 1 thread per device
 *
 *    (mode = SPREAD, places = MACHINE) with any nthreads
 *      -> that will spread threads across all cores of the machine
 */
struct team_binding_t
{
    team_binding_t() :
        mode(XKRT_TEAM_BINDING_MODE_COMPACT),
        places(XKRT_TEAM_BINDING_PLACES_CORE),
        places_list(NULL),
        nplaces(0),
        flags(XKRT_TEAM_BINDING_FLAG_NONE)
    {}

    /* how to distribute threads among places */
    team_binding_mode_t mode;

    /* the places, if XKRT_TEAM_BINDING_PLACES_EXPLICIT - then `team_thread_place_t` must be not null */
    team_binding_places_t places;
    team_thread_place_t * places_list;
    int nplaces;

    /* additional flags */
    team_binding_flag_t flags;

};

/* team description */
struct  team_desc_t
{
    team_desc_t() :
        routine(NULL),
        args(NULL),
        nthreads(0),
        binding()
    {}

    // routine that will be executed by each thread
    team_routine_t routine;

    // user arguments
    void * args;

    // number of threads to spawn
    int nthreads;

    // type of the team
    team_binding_t binding;

    // whether the master thread should be a member of the team or not
    bool master_is_member;
};

/* thread states */
typedef enum    thread_state_t
{
    XKRT_THREAD_UNINITIALIZED   = 0,
    XKRT_THREAD_INITIALIZED     = 1
}               thread_state_t;

/* a team, currently is made of 1 thread max per device, bound onto its closest physical cpu */
struct team_t
{
    // default constructor
    team_t() : desc() {}

    // team description, to be filled by the user before forking it
    team_desc_t desc;

    struct {

        ////////////////////////////////////////////////////////

        /* threads */
        thread_t * threads;

        /* thread states, use for synchronizing */
        thread_state_t * threads_state;

        /* number of threads */
        int nthreads;

        // custom barrier for workstealing
        struct {
            std::atomic<int> n;     /* for spawned threads to sync */
            volatile int version;
            pthread_cond_t cond;    /* to sleep threads when synchronizing */
            pthread_mutex_t mtx;
        } barrier;

        // critical
        struct {
            pthread_mutex_t mtx;
        } critical;

        // groups
        team_hierarchy_t hierarchy;

        ////////////////////////////////////////////////////////////

        // if the routine is parallel for, then this is the stack of lambdas to execute
        # define XKRT_TEAM_PARALLEL_FOR_MAX_FUNC 1
        alignas(hardware_destructive_interference_size)
        struct {
            uint32_t index;
            std::function<void(thread_t * thread)> f[XKRT_TEAM_PARALLEL_FOR_MAX_FUNC];
            uint32_t completed;
            std::atomic<uint32_t> pending;
        } parallel_for;

    } priv;

    /* get a thread */
    thread_t * get_thread(int tid);

    /* get the number of thread of that team */
    inline int
    get_nthreads(void)
    {
        return this->priv.nthreads;
    }

    /* wakeup all threads of the team */
    void wakeup(void);

    /* get iterations */
    static inline void
    parallel_for_thread_bounds(
        int * p_last_iter,
        int * p_lower,
        int * p_upper,
        int incr
    ) {
        thread_t * thread = thread_t::get_tls();
        assert(thread);

        int p_upper_old = *p_upper;
        int trip_count = (incr > 0) ? ((*p_upper - *p_lower) / incr) + 1 : ((*p_lower - *p_upper) / (-incr)) + 1;

        int nthreads = thread->team->priv.nthreads;
        int tid      = thread->tid;

        if (trip_count <= nthreads)
        {
            if (tid < trip_count)
            {
                *p_upper = *p_lower = *p_lower + tid * incr;
                if (p_last_iter)
                    *p_last_iter = (tid == trip_count - 1);
            }
            else
            {
                *p_lower = *p_upper + incr;
                return;
            }
            return ;
        }
        else
        {
            int chunk_size = trip_count / nthreads;
            int extras = trip_count % nthreads;

            if (tid < extras)
            {
                /* The first part is homogeneous with a chunk size a little bit larger */
                *p_upper = *p_lower + (tid + 1) * (chunk_size + 1) * incr - incr;
                *p_lower = *p_lower + tid * (chunk_size + 1) * incr;
            }
            else
            {
                *p_upper = *p_lower + extras * (chunk_size + 1) * incr +
                    (tid + 1 - extras) * chunk_size * incr - incr;
                *p_lower = *p_lower + extras * (chunk_size + 1) * incr +
                    (tid - extras) * chunk_size * incr;
            }

            if (p_last_iter)
            {
                if (incr > 0)
                    *p_last_iter = *p_lower <= p_upper_old && *p_upper > p_upper_old - incr;
                else
                    *p_last_iter = *p_lower >= p_upper_old && *p_upper < p_upper_old - incr;
            }
        }
    }



};

XKRT_NAMESPACE_END

# endif /* __XKRT_TEAM_H__ */
