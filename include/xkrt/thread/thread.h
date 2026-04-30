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

# ifndef __XKRT_THREAD_H__
#  define __XKRT_THREAD_H__

#  include <xkrt/consts.h>
#  include <xkrt/memory/alignas.h>
#  include <xkrt/support.h>
#  include <xkrt/sync/spinlock.h>
#  include <xkrt/task/task.hpp>
// #  include <xkrt/thread/deque.hpp>
#  include <xkrt/thread/naive-deque.hpp>
#  include <xkrt/thread/team-thread-place.h>

#  include <pthread.h>
#  include <atomic>
#  include <random>

#  include <linux/futex.h>      /* Definition of FUTEX_* constants */
#  include <sys/syscall.h>      /* Definition of SYS_* constants */
#  include <unistd.h>

XKRT_NAMESPACE_BEGIN

/////////////
// THREADS //
/////////////

struct team_t;

constexpr task_flag_bitfield_t  XKRT_IMPLICIT_TASK_FLAGS = TASK_FLAG_DOMAIN | TASK_FLAG_GRAPH;
constexpr size_t                XKRT_IMPLICIT_TASK_SIZE  = task_compute_size(XKRT_IMPLICIT_TASK_FLAGS, 0);

/* a thread */
struct alignas(xkrt_pagesize) thread_t
{
    public:

        /* set the current TLS */
        static void push_tls(thread_t * thread);

        /* pop to the previous TLS */
        static void pop_tls(void);

        /* get the TLS */
        static thread_t * get_tls(void);

    public:

        /* the thread team */
        team_t * team;

        /* the place assigned to that thread */
        team_thread_place_t place;

        /* the thread implicit task */
        union {
            task_t implicit_task;
            char _implicit_task_buffer[XKRT_IMPLICIT_TASK_SIZE];
        };

        /* the current task */
        task_t * current_task;

        /* the current task record */
        task_t * current_task_record;

        # if XKRT_SUPPORT_DEBUG
        std::vector<task_t *> tasks;
        # endif /* XKRT_SUPPORT_DEBUG */

        /* the pthread */
        pthread_t pthread;

        /* global thread tid */
        int gtid;

        /* the tid in the team */
        int tid;

        /* the device global id attached to that thread */
        device_unique_id_t device_unique_id;

        /* the thread deque */
        deque_t<task_t *, XKRT_THREAD_DEQUE_CAPACITY> deque;

        /* tasks stack */
        uint8_t * memory_stack_bottom;

        /* next free task pointer in the stack */
        uint8_t * memory_stack_ptr;

        /* memory capacity */
        size_t memory_stack_capacity;

        /* random number generator */
        std::minstd_rand rng;

        /* lock and condition to sleep the mutex */
        struct {
            pthread_mutex_t lock;
            pthread_cond_t  cond;
            volatile bool sleeping;
        } sleep;

        struct {
            /* next function index in the team functions */
            uint32_t index;

        } parallel_for;

        /* previous TLS */
        thread_t * prev;

    public:

        // thread_t(int tid) : thread_t(tid, 0, XKRT_UNSPECIFIED_DEVICE_UNIQUE_ID) {}

        thread_t(
            team_t * team,
            int tid,
            pthread_t pthread,
            device_unique_id_t device_unique_id,
            team_thread_place_t place
        ) :
            team(team),
            place(place),
            implicit_task(XKRT_TASK_FORMAT_NULL, XKRT_IMPLICIT_TASK_FLAGS),
            pthread(pthread),
            gtid(gettid()),
            tid(tid),
            device_unique_id(device_unique_id),
            deque(),
            memory_stack_bottom(NULL),
            memory_stack_capacity(XKRT_THREAD_MAX_MEMORY),
            rng(),
            parallel_for{.index = 0},
            prev(NULL)
        {
            // set current task
            this->current_task = &this->implicit_task;
            this->current_task_record = NULL;

            // initialize sync primitives
            pthread_mutex_init(&this->sleep.lock, 0);
            pthread_cond_init (&this->sleep.cond, 0);
            this->sleep.sleeping = false;

            // initialize implicit task dependency domain
            task_dom_info_t * dom = TASK_DOM_INFO(&this->implicit_task);
            new (dom) task_dom_info_t();
            # if XKRT_SUPPORT_DEBUG
            snprintf(this->implicit_task.label, sizeof(this->implicit_task.label), "implicit");
            # endif

            // initialize memory allocator
            while (1)
            {
                this->memory_stack_bottom = (uint8_t *) malloc(this->memory_stack_capacity);
                if (this->memory_stack_bottom)
                    break ;

                this->memory_stack_capacity = (size_t) (this->memory_stack_capacity * 2 / 3);
                if (this->memory_stack_capacity == 0)
                {
                    this->memory_stack_bottom = NULL;
                    break ;
                }
            }
            this->memory_stack_ptr = this->memory_stack_bottom;
            assert(this->memory_stack_bottom);
        }

        ~thread_t()
        {
            free(this->memory_stack_bottom);
        }

    public:


        /* pause the thread until 'test' returns false */
        template<typename Func>
        inline void
        pause(Func && test)
        {
            // poll a few time before actually taking the lock
            for (int i = 0 ; i < 16 ; ++i)
            {
                if (!test())
                    return ;
            }

            pthread_mutex_lock(&this->sleep.lock);
            {
                while (test())
                {
                    this->sleep.sleeping = true;
                    pthread_cond_wait(&this->sleep.cond, &this->sleep.lock);
                }
            }
            pthread_mutex_unlock(&this->sleep.lock);
        }

        inline void
        wakeup(void)
        {
            pthread_mutex_lock(&this->sleep.lock);
            {
                this->sleep.sleeping = false;
                pthread_cond_signal(&this->sleep.cond);
            }
            pthread_mutex_unlock(&this->sleep.lock);
        }

        void warmup(void);

    /////////////////
    // TASK HELPER //
    /////////////////

    public:

        /**
         * @brief Attempt to steal and execute a task from another thread's queue
         * @return Pointer to the stolen task, or nullptr if no task available
         */
        task_t * worksteal(void);
};

XKRT_NAMESPACE_END

# endif /* __XKRT_THREAD_H__ */
