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

#ifndef __LOGGER_H__
# define __LOGGER_H__

# include <xkrt/support.h>
# include <xkrt/sync/spinlock.h>

# include <time.h>
# include <unistd.h>
# include <stdio.h>
# include <stdlib.h>
# include <stdint.h>

extern spinlock_t LOGGER_PRINT_MTX;

# ifndef LOGGER_FD
#  define LOGGER_FD stderr
# endif

# ifndef LOGGER_HEADER
#  define LOGGER_HEADER "LOGGER"
# endif

# define LOGGER_PRINT_FATAL_ID      0
# define LOGGER_PRINT_ERROR_ID      1
# define LOGGER_PRINT_WARN_ID       2
# define LOGGER_PRINT_INFO_ID       3
# define LOGGER_PRINT_IMPL_ID       4
# define LOGGER_PRINT_DEBUG_ID      5

extern char * LOGGER_PRINT_COLORS[6];
extern char * LOGGER_PRINT_HEADERS[6];

extern int LOGGER_VERBOSE;
extern int LOGGER_INITIALIZED;

extern volatile double   LOGGER_TIME_ELAPSED;
extern volatile uint64_t LOGGER_LAST_TIME;

# define LOGGER_PRINT_LINE() \
    fprintf(LOGGER_FD, "%s:%d (%s)\n", __FILE__, __LINE__, __func__);

# if LOGGER_SHUT_UP

#  define LOGGER_PRINT(...)
#  define LOGGER_NOT_IMPLEMENTED_WARN(...)
#  define LOGGER_NOT_SUPPORTED(...)
#  define LOGGER_NOT_IMPLEMENTED(...)
#  define LOGGER_INFO(...)
#  define LOGGER_WARN(...)
#  define LOGGER_ERROR(...)
#  define LOGGER_IMPL(...)
#  define LOGGER_DEBUG(...)
#  define LOGGER_FATAL(...)                                                         \
    do {                                                                            \
        fprintf(LOGGER_FD, "XKRT aborted with the following error message:\n\t");   \
        fprintf(LOGGER_FD, __VA_ARGS__);                                            \
        fprintf(LOGGER_FD, "\n");                                                   \
        fprintf(LOGGER_FD, "at");                                                   \
        LOGGER_PRINT_LINE();                                                        \
        fflush(LOGGER_FD);                                                          \
        abort();                                                                    \
    } while (0)

# else /* LOGGER_SHUT_UP */

# define LOGGER_PRINT(LVL, ...)                                                 \
    do {                                                                        \
        if (!LOGGER_INITIALIZED)                                                \
        {                                                                       \
            LOGGER_INITIALIZED = 1;                                             \
            char * s = getenv("LOGGER_VERBOSE");                                \
            if (s)                                                              \
                LOGGER_VERBOSE = atoi(s);                                       \
        }                                                                       \
        if (LVL <= LOGGER_VERBOSE)                                              \
        {                                                                       \
            SPINLOCK_LOCK(LOGGER_PRINT_MTX);                                    \
            struct timespec _ts;                                                \
            clock_gettime(CLOCK_MONOTONIC, &_ts);                               \
            uint64_t t = (uint64_t)(_ts.tv_sec * 1000000000) +                  \
                            (uint64_t) _ts.tv_nsec;                             \
            if (LOGGER_LAST_TIME != 0)                                          \
                LOGGER_TIME_ELAPSED = LOGGER_TIME_ELAPSED +                     \
                                        (double) (t - LOGGER_LAST_TIME) / 1e9;  \
            LOGGER_LAST_TIME = t;                                               \
            if (isatty(STDOUT_FILENO))                                          \
                fprintf(LOGGER_FD, "[%8lf] "                                    \
                                "[TID=%d] "                                     \
                                "[\033[1;37m" LOGGER_HEADER "\033[0m] "         \
                                "[%s%s\033[0m] ",                               \
                                LOGGER_TIME_ELAPSED,                            \
                                gettid(),                                       \
                                LOGGER_PRINT_COLORS[LVL],                       \
                                LOGGER_PRINT_HEADERS[LVL]);                     \
            else                                                                \
                fprintf(LOGGER_FD, "[%8lf]"                                     \
                                "[TID=%d] "                                     \
                                "[" LOGGER_HEADER "] "                          \
                                "[%s] ",                                        \
                                LOGGER_TIME_ELAPSED,                            \
                                gettid(),                                       \
                                LOGGER_PRINT_HEADERS[LVL]);                     \
            fprintf(LOGGER_FD, __VA_ARGS__);                                    \
            fprintf(LOGGER_FD, "\n");                                           \
            fflush(LOGGER_FD);                                                  \
            SPINLOCK_UNLOCK(LOGGER_PRINT_MTX);                                  \
        }                                                                       \
        if (LVL == LOGGER_PRINT_FATAL_ID)                                       \
        {                                                                       \
            LOGGER_PRINT_LINE();                                                \
            fflush(LOGGER_FD);                                                  \
            abort();                                                            \
        }                                                                       \
    } while (0)

# define LOGGER_NOT_IMPLEMENTED_WARN(S)                                 \
    LOGGER_IMPL("'%s' at %s:%d in %s()",                                \
            S, __FILE__, __LINE__, __func__);

# define LOGGER_NOT_SUPPORTED()   LOGGER_NOT_IMPLEMENTED_WARN("Not supported")
# define LOGGER_NOT_IMPLEMENTED() LOGGER_NOT_IMPLEMENTED_WARN("Not implemented")

# define LOGGER_INFO(...)  LOGGER_PRINT(LOGGER_PRINT_INFO_ID,  __VA_ARGS__)
# define LOGGER_WARN(...)  LOGGER_PRINT(LOGGER_PRINT_WARN_ID,  __VA_ARGS__)
# define LOGGER_ERROR(...) LOGGER_PRINT(LOGGER_PRINT_ERROR_ID, __VA_ARGS__)
# define LOGGER_IMPL(...)  LOGGER_PRINT(LOGGER_PRINT_IMPL_ID,  __VA_ARGS__)
# define LOGGER_FATAL(...) LOGGER_PRINT(LOGGER_PRINT_FATAL_ID, __VA_ARGS__)
# if XKRT_SUPPORT_DEBUG
#  define LOGGER_DEBUG(...) LOGGER_PRINT(LOGGER_PRINT_DEBUG_ID, __VA_ARGS__)
# else
#  define LOGGER_DEBUG(...)
# endif

# endif /* LOGGER_SHUT_UP */

#endif /* __LOGGER_H__*/
