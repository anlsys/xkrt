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

#ifndef __STATS_H__
# define __STATS_H__

# include <xkrt/support.h>

# if XKRT_SUPPORT_STATS

# include <atomic>
# include <stddef.h>

typedef std::atomic<uint64_t> stats_int_t;

/* per-format task counters, pool-allocated (one entry per registered format) */
typedef struct  task_stats_t
{
    stats_int_t commited;       ///< Number of tasks committed
    stats_int_t completed;      ///< Number of tasks completed
}               task_stats_t;

#  define XKRT_STATS_INCR(X, INCR)  \
    do {                            \
        X += INCR;                  \
    } while (0)

/* per-format counter increment (the slot is created alongside the format) */
#  define XKRT_STATS_TASK_INCR(STATS, FMTID, FIELD, INCR)   \
    do {                                                    \
        (STATS).tasks.get(FMTID)->FIELD += (INCR);          \
    } while (0)

#  define XKRT_STATS_DECR(X, DECR)  \
    do {                            \
        X -= DECR;                  \
    } while (0)

#  define XKRT_STATS_SET(X, V)      \
    do {                            \
        X = V;                      \
    } while (0)

# else /* XKRT_SUPPORT_STATS */

#  define XKRT_STATS_INCR(X, INCR)
#  define XKRT_STATS_TASK_INCR(STATS, FMTID, FIELD, INCR)
#  define XKRT_STATS_DECR(X, DECR)
#  define XKRT_STATS_SET(X, V)

# endif /* XKRT_SUPPORT_STATS */

#endif /* __STATS_H__ */
