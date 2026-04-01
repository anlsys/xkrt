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

#ifndef __DRIVER_HOST_H__
# define __DRIVER_HOST_H__

# include <atomic>

# include <xkrt/driver/driver.h>
# include <xkrt/driver/queue.h>

# include <linux/io_uring.h>

XKRT_NAMESPACE_BEGIN

typedef std::atomic<uint8_t> queue_host_event_t;

typedef struct  queue_host_t
{
    command_queue_t super;

    /* async i/o */
    struct {

        /* io_uring file desc */
        int fd;

        /* submission queue */
        void * sq_ptr;
        unsigned * sq_tail;
        unsigned * sq_mask;
        unsigned * sq_array;

        struct io_uring_sqe * sqes;

        /* completion queue */
        void * cq_ptr;
        unsigned * cq_head;
        unsigned * cq_tail;
        unsigned * cq_mask;

        struct io_uring_cqe * cqes;

        /* count of SQEs queued but not submitted */
        unsigned  pending_submits;

        /* pointer to SQ flags word in shared ring */
        unsigned * sq_flags;

    } io_uring;
}               queue_host_t;

typedef struct  driver_host_t
{
    driver_t super;
}               driver_host_t;

XKRT_NAMESPACE_END

#endif /* __DRIVER_HOST_H__ */
