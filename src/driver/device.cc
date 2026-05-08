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

# include <xkrt/driver/device.hpp>
# include <xkrt/thread/team.h>

XKRT_NAMESPACE_USE;

//////////////////////
// MEMORY MANAGMENT //
//////////////////////

void
device_t::memory_reset_on(int area_idx)
{
    assert(this->allocator);
    this->allocator->reset_on(area_idx);

    XKRT_STATS_INCR(this->stats.memory.freed, this->stats.memory.allocated.currently);
    XKRT_STATS_SET (this->stats.memory.allocated.currently, 0);
}

void
device_t::memory_reset(void)
{
    assert(this->allocator);
    this->allocator->reset();

    XKRT_STATS_INCR(this->stats.memory.freed, this->stats.memory.allocated.currently);
    XKRT_STATS_SET (this->stats.memory.allocated.currently, 0);
}

void
device_t::memory_deallocate(area_chunk_t * chunk)
{
    return this->memory_deallocate_on(chunk, chunk->area_idx);
}

void
device_t::memory_deallocate_on(area_chunk_t * chunk, int area_idx)
{
    assert(chunk->area_idx >= 0);
    assert(chunk->area_idx < this->nmemories);
    assert(this->allocator);

    size_t chunk_size = chunk->size;
    this->allocator->deallocate_on(chunk, area_idx);

    XKRT_STATS_INCR(this->stats.memory.freed, chunk_size);
    XKRT_STATS_DECR(this->stats.memory.allocated.currently, chunk_size);
}

area_chunk_t *
device_t::memory_allocate_on(const size_t user_size, int area_idx)
{
    assert(area_idx >= 0);
    assert(area_idx < this->nmemories);
    assert(this->allocator);

    const size_t size = (user_size + 7UL) & ~7UL;
    area_chunk_t * curr = this->allocator->allocate_on(user_size, area_idx);

    if (curr)
    {
        XKRT_STATS_INCR(this->stats.memory.allocated.total,       size);
        XKRT_STATS_INCR(this->stats.memory.allocated.currently,   size);
    }

    return curr;
}

area_chunk_t *
device_t::memory_allocate(const size_t user_size)
{
    return this->memory_allocate_on(user_size, 0);
}

///////////////////////
// QUEUE MANAGEMENT //
///////////////////////

void
device_t::offloader_queues_are_empty(
    int tid,
    const command_queue_type_t qtype,
    bool * ready,
    bool * pending
) const {

    *ready   = false;
    *pending = false;

    unsigned int bgn = (qtype == XKRT_QUEUE_TYPE_ALL) ?                    0 : qtype;
    unsigned int end = (qtype == XKRT_QUEUE_TYPE_ALL) ? XKRT_QUEUE_TYPE_ALL : qtype + 1;

    for (unsigned int s = bgn ; s < end ; ++s)
    {
        for (int i = 0 ; i < this->count[s] ; ++i)
        {
            const command_queue_t * queue = this->queues[tid][s][i];
            if (*ready == false && !queue->ready.is_empty())
                *ready = true;
            if (*pending == false && !queue->pending.is_empty())
                *pending = true;
            if (*ready && *pending)
                return ;
        }
    }
}

void
device_t::offloader_queue_next(
    command_queue_type_t qtype,
    thread_t ** pthread,    /* OUT */
    command_queue_t ** pqueue       /* OUT */
) {
    // round robin on the thread for this queue type
    int next_thread = this->next_thread.fetch_add(1, std::memory_order_relaxed) % this->team->get_nthreads();

    // round robin on queues for the queues of the given type on the choosen thread
    int count = this->count[qtype];
    assert(count);
    int snext = this->next_queue[next_thread][qtype].fetch_add(1, std::memory_order_relaxed) % count;

    // save thread/queue
    *pthread = this->team->get_thread(next_thread);
    *pqueue  = this->queues[next_thread][qtype][snext];
}

////////////////////////
// COMMAND SUBMISSION //
////////////////////////

void
device_t::offloader_queue_command_new(
    const command_queue_type_t qtype,   /* IN  */
    const ocg::command_type_t ctype,         /* IN  */
    const command_flag_t flags,         /* IN  */
    thread_t ** pthread,                /* OUT */
    command_queue_t ** pqueue,          /* OUT */
    command_t ** pcommand                   /* OUT */
) {
    assert(pqueue);
    assert(pcommand);

    /* retrieve native queue */
    this->offloader_queue_next(qtype, pthread, pqueue);
    assert(*pthread);
    assert(*pqueue);
    assert((*pqueue)->type == qtype);

    /* allocate the command */
    do {
        SPINLOCK_LOCK((*pqueue)->spinlock);
        (*pcommand) = (*pqueue)->command_new(ctype, flags);
        if (*pcommand)
            break ; /* will be unlock during 'commit' */
        SPINLOCK_UNLOCK((*pqueue)->spinlock);
        LOGGER_FATAL("Stream is full, increase 'XKRT_OFFLOADER_CAPACITY' or implement support for full-queue management yourself :-) (sorry)");
    } while (1);
}

/* commit a queue command and wakeup thread */
void
device_t::offloader_queue_command_commit(
    thread_t * thread,          /* thread that will execute the command */
    command_queue_t * queue,    /* queue of that thread */
    command_t * command             /* the command */
) {
    /* If the current task is recording */
    thread_t * tls = thread_t::get_tls();
    assert(tls);

    /* If recording */
    task_t * task = tls->current_task_record;
    if (task)
    {
        if (!(command->flags & COMMAND_FLAG_PROG_LAUNCHER))
        {
            assert(task->flags & TASK_FLAG_RECORD);
            assert(
                task->state.value == TASK_STATE_DATA_FETCHING ||  // emitted for fetching data before making the task ready
                task->state.value == TASK_STATE_EXECUTING     ||  // emitted alongside execution
                task->state.value == TASK_STATE_COMPLETED         // prefetching
            );
            assert(task->parent);
            assert(task->parent->flags & TASK_FLAG_GRAPH);

            command_t * commandrec = task_put_command_record(task);
            memcpy(commandrec, command, sizeof(command_t));
            commandrec->completion_callback_clear();

            // if skipping command execution
            if (!(task->parent->flags & TASK_FLAG_GRAPH_EXECUTE_COMMAND))
            {
                // complete it now and return
                command->completion_callback_raise();
                SPINLOCK_UNLOCK(queue->spinlock);
                return ;
            }
        }
    }

    /* commit command to the queue */
    queue->commit(command);
    SPINLOCK_UNLOCK(queue->spinlock);

    /* wakeup device worker thread */
    thread->wakeup();
}
