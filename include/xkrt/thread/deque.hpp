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

# ifndef __XKRT_LOCKFREE_DEQUE_HPP__
#  define __XKRT_LOCKFREE_DEQUE_HPP__

#  include <xkrt/memory/alignas.h>
#  include <xkrt/sync/spinlock.h>

#  include <atomic>
#  include <new>

XKRT_NAMESPACE_BEGIN

/**
 *  This data structure is a deque for multi-threading workstealing.
 *      - T is the storage type (e.g., task_t *)
 *      - C is the capacity of the internal storage
 *
 *  Let be a team of threads where each thread has its own deque.
 *  When a thread produces a task, it pushes to the end of its own deque using `push`.
 *  When a thread is looking for work, it in-order
 *      - try to pop a single task from the end of its own deque (`pop`)
 *      - if not, it will find another deque 'D' of another thread, and call successively
 *          - D.steal(&task_array, &n) -- that will steal 'n' of tasks in 'D' ('n' should be about half the size of 'D')
 */
template<typename T, int C>
struct deque_t
{
    /* Storage */
    T tasks[C];

    /* A lock when needs be */
    alignas(hardware_destructive_interference_size) spinlock_t lock;

    /* Head of the deque */
    alignas(hardware_destructive_interference_size) std::atomic<int> _h;

    /* Tail of the deque */
    alignas(hardware_destructive_interference_size) std::atomic<int> _t;

    /* Default constructor / destructor */
    deque_t() : tasks{}, lock(0), _h(0), _t(0) {}
    ~deque_t() {}

    /**
     *  Push a single element to the tail -- from the thread that own the deque
     *  Return 0 on success, 1 on failure (if queue is full)
     */
    int push(T const & t);

    /**
     *  Try to push at most 'n' elements to the tail.
     *  Return the number 'm' <= 'n' so that elements ts[0:m] got pushed to the tail
     */
    int push(T const * ts, int n);

    /**
     *  Push a single element to the head -- from thread that do not own the deque
     *  Return 0 on success, 1 on failure (if queue is full)
     */
    int give(T const & t);

    /**
     *  Remove and return a copy of the tail.
     */
    T pop(void);

    /**
     *  Steal tasks.
     *  Return a pointer in 'tasks' of the first element and the number of elements to steal 'n'
     */
    int steal(T ** ts, int * n);

    /**
     *  Finalize steal request completion, typically after 'ts[0:*n]' tasks had
     *  been copied to the deque of the thread stealing.
     */
    void stolen(T ** ts, int * n);
};

XKRT_NAMESPACE_END

# endif /* __XKRT_LOCKFREE_DEQUE_HPP__ */
