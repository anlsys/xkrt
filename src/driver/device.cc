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
    area_t * area = &(this->memories[area_idx].area);

    # pragma message(TODO "This is leaking")
    area_chunk_t * chunk0 = (area_chunk_t *) malloc(sizeof(area_chunk_t));
    assert(chunk0);
    memcpy(chunk0, &(area->chunk0), sizeof(area_chunk_t));
    area->free_chunk_list = chunk0;

    XKRT_STATS_INCR(this->stats.memory.freed, this->stats.memory.allocated.currently);
    XKRT_STATS_SET (this->stats.memory.allocated.currently, 0);
}

void
device_t::memory_reset(void)
{
    for (int i = 0 ; i < this->nmemories ; ++i)
        this->memory_reset_on(i);
}

void
device_t::memory_set_chunk0(
    uintptr_t ptr,
    size_t size,
    int area_idx
) {
    area_t * area = &(this->memories[area_idx].area);

    area->chunk0.ptr            = ptr;
    area->chunk0.size           = size;
    area->chunk0.state          = XKRT_ALLOC_CHUNK_STATE_FREE;
    area->chunk0.prev           = NULL;
    area->chunk0.next           = NULL;
    area->chunk0.freelink       = NULL;
    area->chunk0.use_counter    = 0;

    this->memory_reset_on(area_idx);
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
    area_t * area = &(this->memories[area_idx].area);

    bool delete_chunk = false;
    XKRT_MUTEX_LOCK(area->lock);
    {
        chunk->state = XKRT_ALLOC_CHUNK_STATE_FREE;
        chunk->use_counter = 0;

        /* can we merge chunk into next_chunk ? */
        area_chunk_t * next_chunk = chunk->next;
        if (next_chunk && next_chunk->state == XKRT_ALLOC_CHUNK_STATE_FREE)
        {
            next_chunk->prev = chunk->prev;
            if (chunk->prev)
                chunk->prev->next = next_chunk;
            next_chunk->size += chunk->size;
            assert(next_chunk->ptr > chunk->ptr);
            next_chunk->ptr = chunk->ptr;
            delete_chunk = true;
        }

        area_chunk_t * prev_chunk = chunk->prev;
        if (prev_chunk)
        {
            /*  if prev_chunk is a free chunk and 'delete_chunk' is true,
             *  then we have to merge prev and next */
            if (prev_chunk->state == XKRT_ALLOC_CHUNK_STATE_FREE)
            {
                if (delete_chunk)
                {
                    assert(prev_chunk->ptr < chunk->ptr);
                    assert(prev_chunk->ptr < next_chunk->ptr);

                    prev_chunk->size += next_chunk->size;
                    prev_chunk->next = next_chunk->next;
                    if (next_chunk->next)
                        next_chunk->next->prev = prev_chunk;
                    prev_chunk->freelink = next_chunk->freelink;
                    free(next_chunk);
                }
                else
                {
                    /* merge chunk into prev_chunk */
                    assert(prev_chunk->ptr < chunk->ptr);
                    prev_chunk->next = chunk->next;
                    if (chunk->next)
                        chunk->next->prev = prev_chunk;
                    prev_chunk->size += chunk->size;
                    delete_chunk = true;
                }
            }
            else if (!delete_chunk)
            {
                /* free_chunk_list is ordered by increasing adress: search form prev the previous bloc */
                while (prev_chunk && prev_chunk->state != XKRT_ALLOC_CHUNK_STATE_FREE)
                    prev_chunk = prev_chunk->prev;

                if (!prev_chunk)
                {
                    chunk->freelink = area->free_chunk_list;
                    area->free_chunk_list = chunk;
                }
                else
                {
                    chunk->freelink = prev_chunk->freelink;
                    prev_chunk->freelink = chunk;
                }
            }
        }
        else if (!delete_chunk)
        {
            chunk->freelink = area->free_chunk_list;
            area->free_chunk_list = chunk;
        }
    }
    XKRT_MUTEX_UNLOCK(area->lock);

    XKRT_STATS_INCR(this->stats.memory.freed, chunk->size);
    XKRT_STATS_DECR(this->stats.memory.allocated.currently, chunk->size);

    if (delete_chunk)
        free(chunk);
}

area_chunk_t *
device_t::memory_allocate_on(const size_t user_size, int area_idx)
{
    /* retrieve area */
    assert(area_idx >= 0);
    assert(area_idx < this->nmemories);
    area_t * area = &(this->memories[area_idx].area);

    /* align data */
    const size_t size = (user_size + 7UL) & ~7UL;
    area_chunk_t * curr;

    XKRT_MUTEX_LOCK(area->lock);
    {
        /* best fit strategy */
        curr = area->free_chunk_list;

        area_chunk_t * prevfree = NULL;
        size_t min_size = 0;
        area_chunk_t * min_size_curr = NULL;
        area_chunk_t * min_size_prevfree = NULL;

        while (curr)
        {
            size_t curr_size = curr->size;
            if (curr_size >= size)
            {
                if ((min_size_curr == 0) || (min_size > curr_size))
                {
                    min_size = curr_size;
                    min_size_curr = curr;
                    min_size_prevfree = prevfree;
                }
            }
            prevfree = curr;
            curr = curr->freelink;
        }

        /* and the winner is min_size_curr ! */
        curr = min_size_curr;
        prevfree = min_size_prevfree;

        /* split chunk */
        if ((curr != NULL) && (min_size - size >= (size_t)(0.5*(double)size)))
        {
            size_t curr_size = curr->size;
            area_chunk_t * remainder = (area_chunk_t *) malloc(sizeof(area_chunk_t));
            remainder->ptr         = size + curr->ptr;
            remainder->size        = (curr_size - size);
            remainder->state       = XKRT_ALLOC_CHUNK_STATE_FREE;
            remainder->use_counter = 0;
            remainder->prev        = curr;
            remainder->next        = curr->next;
            remainder->freelink    = curr->freelink;

            /* link remainder segment after curr */
            if (curr->next)
                curr->next->prev = remainder;
            curr->next = remainder;
            curr->size = size;
            curr->freelink = remainder;
        }

        if (curr != NULL)
        {
            if (prevfree)
                prevfree->freelink = curr->freelink;
            else
                area->free_chunk_list = curr->freelink;
            curr->state = XKRT_ALLOC_CHUNK_STATE_ALLOCATED;
            curr->freelink = NULL;
        }
    }

    XKRT_MUTEX_UNLOCK(area->lock);

    if (curr)
    {
        curr->area_idx = area_idx;
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
    command_t ** pcmd                   /* OUT */
) {
    assert(pqueue);
    assert(pcmd);

    /* retrieve native queue */
    this->offloader_queue_next(qtype, pthread, pqueue);
    assert(*pthread);
    assert(*pqueue);
    assert((*pqueue)->type == qtype);

    /* allocate the command */
    do {
        SPINLOCK_LOCK((*pqueue)->spinlock);
        (*pcmd) = (*pqueue)->command_new(ctype, flags);
        if (*pcmd)
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
    command_t * cmd             /* the command */
) {
    /* If the current task is recording */
    thread_t * tls = thread_t::get_tls();
    assert(tls);

    /* If recording */
    task_t * task = tls->current_task_record;
    if (task)
    {
        if (!(cmd->flags & COMMAND_FLAG_PROG_LAUNCHER))
        {
            assert(task->flags & TASK_FLAG_RECORD);
            assert(
                task->state.value == TASK_STATE_DATA_FETCHING ||  // emitted for fetching data before making the task ready
                task->state.value == TASK_STATE_EXECUTING     ||  // emitted alongside execution
                task->state.value == TASK_STATE_COMPLETED         // prefetching
            );
            assert(task->parent);
            assert(task->parent->flags & TASK_FLAG_GRAPH);

            command_t * cmdrec = task_put_command_record(task);
            memcpy(cmdrec, cmd, sizeof(command_t));
            cmdrec->completion_callback_clear();

            // if skipping command execution
            if (!(task->parent->flags & TASK_FLAG_GRAPH_EXECUTE_COMMAND))
            {
                // complete it now and return
                cmd->completion_callback_raise();
                SPINLOCK_UNLOCK(queue->spinlock);
                return ;
            }
        }
    }

    /* commit command to the queue */
    queue->commit(cmd);
    SPINLOCK_UNLOCK(queue->spinlock);

    /* wakeup device worker thread */
    thread->wakeup();
}

static inline command_queue_type_t
command_type_to_queue_type(
    ocg::command_type_t ctype
) {
    switch (ctype)
    {
        case (ocg::COMMAND_TYPE_PROG):
        case (ocg::COMMAND_TYPE_BATCH):
            return XKRT_QUEUE_TYPE_KERN;

        case (ocg::COMMAND_TYPE_COPY_H2H_1D):
        case (ocg::COMMAND_TYPE_COPY_H2H_2D):
            return XKRT_QUEUE_TYPE_H2D;

        case (ocg::COMMAND_TYPE_COPY_H2D_1D):
        case (ocg::COMMAND_TYPE_COPY_H2D_2D):
            return XKRT_QUEUE_TYPE_H2D;

        case (ocg::COMMAND_TYPE_COPY_D2H_1D):
        case (ocg::COMMAND_TYPE_COPY_D2H_2D):
            return XKRT_QUEUE_TYPE_D2H;

        case (ocg::COMMAND_TYPE_COPY_D2D_1D):
        case (ocg::COMMAND_TYPE_COPY_D2D_2D):
            return XKRT_QUEUE_TYPE_D2D;

        case (ocg::COMMAND_TYPE_FD_READ):
            return XKRT_QUEUE_TYPE_FD_READ;

        case (ocg::COMMAND_TYPE_FD_WRITE):
            return XKRT_QUEUE_TYPE_FD_WRITE;

        case (ocg::COMMAND_TYPE_EMPTY):
        default:
            LOGGER_FATAL("I don't know what queue I should use for that command!");
            return XKRT_QUEUE_TYPE_ALL;
    }
}

int
device_t::command_submit(command_t * cmd)
{
    // command is serialized: it must be launched serially on the calling
    // thread, which returns on the command completion
    if (cmd->flags & COMMAND_FLAG_SERIALIZED)
    {
        assert(cmd->type == ocg::COMMAND_TYPE_EMPTY || cmd->type == ocg::COMMAND_TYPE_PROG);
        assert(cmd->flags & COMMAND_FLAG_SYNCHRONOUS);
        if (cmd->type == ocg::COMMAND_TYPE_EMPTY)
            cmd->completion_callback_raise();
        else if (cmd->type == ocg::COMMAND_TYPE_PROG)
        {
            if (this->unique_id == XKRT_HOST_DEVICE_UNIQUE_ID)
                cmd->prog.launcher.fixed.fn(cmd->prog.launcher.fixed.args);
            else
                LOGGER_FATAL("Unsupported command");
        }
    }
    // else, push to a queue
    else
    {
        command_queue_type_t qtype = command_type_to_queue_type(cmd->type);

        thread_t * thread;
        command_queue_t * queue;
        this->offloader_queue_next(qtype, &thread, &queue);
        assert(thread);
        assert(queue);

        SPINLOCK_LOCK(queue->spinlock);
        queue->emplace(cmd);
        queue->commit(cmd);
        SPINLOCK_UNLOCK(queue->spinlock);

        thread->wakeup();
    }

    return 0;
}
