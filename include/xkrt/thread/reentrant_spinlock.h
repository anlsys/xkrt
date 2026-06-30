/*
** Copyright 2024,2025 INRIA
**
** Contributors :
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

#ifndef __REENTRANT_SPINLOCK_H__
# define __REENTRANT_SPINLOCK_H__

# include <xkrt/namespace.h>
# include <xkrt/sync/mem.h>
# include <xkrt/thread/thread.h>

# define REENTRANT_SPINLOCK_NO_OWNER (-1)

typedef struct  reentrant_spinlock_t
{
    volatile int owner;
    int depth;
}               reentrant_spinlock_t;

# define REENTRANT_SPINLOCK_INITIALIZER { REENTRANT_SPINLOCK_NO_OWNER, 0 }

# define REENTRANT_SPINLOCK_LOCK(L)                                             \
    do {                                                                        \
        const int xkrt_tid_ = XKRT_NAMESPACE::thread_t::get_tls()->tid;         \
        if ((L).owner != xkrt_tid_)                                             \
            while (__sync_val_compare_and_swap(&(L).owner,                      \
                        REENTRANT_SPINLOCK_NO_OWNER, xkrt_tid_)                 \
                    != REENTRANT_SPINLOCK_NO_OWNER)                            \
                mem_pause();                                                    \
        ++(L).depth;                                                            \
    } while (0)

# define REENTRANT_SPINLOCK_TRYLOCK(L)                                          \
    (((L).owner == XKRT_NAMESPACE::thread_t::get_tls()->tid)                    \
        ? (++(L).depth, true)                                                   \
        : ((__sync_val_compare_and_swap(&(L).owner,                             \
                REENTRANT_SPINLOCK_NO_OWNER,                                    \
                XKRT_NAMESPACE::thread_t::get_tls()->tid)                       \
                    == REENTRANT_SPINLOCK_NO_OWNER)                             \
            ? ((L).depth = 1, true)                                             \
            : false))

# define REENTRANT_SPINLOCK_UNLOCK(L)                                           \
    do {                                                                        \
        if (--(L).depth == 0)                                                   \
        {                                                                       \
            mem_barrier();                                                      \
            (L).owner = REENTRANT_SPINLOCK_NO_OWNER;                            \
        }                                                                       \
    } while (0)

# define REENTRANT_SPINLOCK_IS_LOCKED(L)                                        \
    ((L).owner != REENTRANT_SPINLOCK_NO_OWNER)

# define REENTRANT_SPINLOCK_OWNS(L)                                             \
    ((L).owner == XKRT_NAMESPACE::thread_t::get_tls()->tid)

#endif /* __REENTRANT_SPINLOCK_H__ */
