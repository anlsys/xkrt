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

#ifndef __SPINLOCK_H__
# define __SPINLOCK_H__

# include <xkrt/sync/mem.h>

#if 1

typedef volatile char spinlock_t;

# define SPINLOCK_LOCK(L)                                                       \
    do {                                                                        \
        while (__sync_val_compare_and_swap(&L, (char)0, (char)1) == (char) 1)   \
            mem_pause();                                                        \
    } while (0)

/* Evaluates to true if the lock is taken */
# define SPINLOCK_TRYLOCK(L)                                            \
    (__sync_val_compare_and_swap(&L, (char)0, (char)1) == (char) 0)

# define SPINLOCK_UNLOCK(L)             \
    do {                                \
        __sync_fetch_and_xor(&L, L);    \
    } while (0)

# define SPINLOCK_INITIALIZER 0

# else

# include <pthread.h>
typedef pthread_mutex_t spinlock_t;
# define SPINLOCK_LOCK(L)       pthread_mutex_lock(&L)
# define SPINLOCK_UNLOCK(L)     pthread_mutex_unlock(&L)
# define SPINLOCK_INITIALIZER   PTHREAD_MUTEX_INITIALIZER

# endif

#endif /* __SPINLOCK_H__ */
