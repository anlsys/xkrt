/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
** Joao Lima joao.lima@inf.ufsm.br
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

# include <xkrt/task/task.hpp>
# include <xkrt/thread/deque.hpp>
# include <xkrt/logger/logger.h>
# include <xkrt/sync/mem.h>

XKRT_NAMESPACE_BEGIN

// TODO : PROBLEM - this impl assumes `push` and `pop` are called from the same
// 'producer' thread and `steal` from any other thread.
// It does not support 'giving' tasks, that is, having a thread different from
// the producer pushing a task into this queue

template<typename T, int C>
int
deque_t<T, C>::push(T const & t)
{
    int h = _h.load(std::memory_order_relaxed);
    int t_idx = _t.load(std::memory_order_relaxed);

    if (t_idx - h >= C)
        return 1;   /* Full */

    tasks[t_idx % C] = t;
    // Release ensures visibility before the tail pointer increments
    _t.store(t_idx + 1, std::memory_order_release);
    return 0;
}

template<typename T, int C>
int
deque_t<T, C>::push(
    T const * ts,
    int n
) {
    int h = _h.load(std::memory_order_relaxed);
    int t_idx = _t.load(std::memory_order_relaxed);

    int available = C - (t_idx - h);
    int m = std::min(n, available);

    for (int i = 0; i < m; ++i)
        tasks[(t_idx + i) % C] = ts[i];

    _t.store(t_idx + m, std::memory_order_release);
    return m;
}

template<typename T, int C>
T
deque_t<T, C>::pop(void)
{
    int t_idx = _t.load(std::memory_order_relaxed) - 1;
    _t.store(t_idx, std::memory_order_relaxed);

    // Ensure the store to _t is visible before loading _h
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int h = _h.load(std::memory_order_relaxed);

    T result = NULL;

    if (h <= t_idx)
    {
        result = tasks[t_idx % C];
        if (h == t_idx) {
            // Potential race with a thief
            SPINLOCK_LOCK(lock);
            if (_h.load(std::memory_order_relaxed) > t_idx)
            {
                // Thief won
                result = T{};
            }
            else
            {
                // Owner won, advance head to signify "empty"
                _h.fetch_add(1, std::memory_order_relaxed);
            }
            SPINLOCK_UNLOCK(lock);
            _t.store(t_idx + 1, std::memory_order_relaxed);
        }
    }
    else
    {
        // Deque was already empty
        _t.store(t_idx + 1, std::memory_order_relaxed);
    }
    return result;
}

template<typename T, int C>
int
deque_t<T, C>::steal(
    T ** ts,
    int * n
) {
    SPINLOCK_LOCK(lock);

    // Acquire ensures we see the latest pushes from the owner
    int h = _h.load(std::memory_order_acquire);
    int t = _t.load(std::memory_order_acquire);

    int size = t - h;
    if (size <= 0)
    {
        SPINLOCK_UNLOCK(lock);
        *n = 0;
        return 1; // Fail
    }

    // Standard work-stealing: steal half the tasks
    *n = std::max(1, size / 2);

    // Note: In a circular buffer, a batch steal might wrap around.
    // For simplicity, we limit 'n' to the contiguous segment.
    int till_end = C - (h % C);
    if (*n > till_end) *n = till_end;

    *ts = &tasks[h % C];

    // We keep the lock held until 'stolen' is called to prevent
    // the owner or other thieves from invalidating this memory.
    return 0;
}

template<typename T, int C>
void
deque_t<T, C>::stolen(
    T ** ts,
    int * n
) {
    // Advance head to finalize the removal of 'n' tasks
    _h.fetch_add(*n, std::memory_order_release);
    SPINLOCK_UNLOCK(lock);
}

template<typename T, int C>
int
deque_t<T, C>::give(T const & t)
{
    // We must lock because we are competing with thieves (and other givers) at the head.
    SPINLOCK_LOCK(lock);

    int h = _h.load(std::memory_order_relaxed);
    int t_idx = _t.load(std::memory_order_relaxed);

    // Check for capacity
    if (t_idx - h >= C)
    {
        SPINLOCK_UNLOCK(lock);
        return 1; // Queue is full
    }

    // Calculate the new head position (prepended)
    int new_h = h - 1;

    // Circular buffer index: ((index % C) + C) % C handles negative results of %
    int slot = ((new_h % C) + C) % C;
    tasks[slot] = t;

    // Release fence ensures the task is fully written before a thief can see the new head index
    _h.store(new_h, std::memory_order_release);

    SPINLOCK_UNLOCK(lock);
    return 0;
}

// Explicit instantiation
template struct deque_t<task_t *, XKRT_THREAD_DEQUE_CAPACITY>;

XKRT_NAMESPACE_END
