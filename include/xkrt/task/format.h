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

/**
 * @file xkrt_task_format.h
 * @brief Defines structures and functions for managing task formats and their targets.
 *
 * This header file provides the definitions for different task execution targets
 * (like HOST, CUDA, ZE, etc.), structures to hold task functions for these
 * targets, and mechanisms to manage a collection of task formats.
 */

#ifndef __XKRT_TASK_FORMAT_H__
# define __XKRT_TASK_FORMAT_H__

# include <stdint.h>

/* Shared (C-compatible) source-code descriptor, also used by CGIR's
 * `command_prog_t`. See `cgir/prog-source.h`. */
# include <cgir/prog-source.h>

/**
 * @enum xkrt_task_format_target_t
 * @brief Enumerates the possible execution targets for a task.
 *
 * This enumeration defines the different types of hardware or execution
 * environments where a task can be launched.
 */
typedef enum    xkrt_task_format_target_t
{
    XKRT_TASK_FORMAT_TARGET_HOST  = 0, /**< CPU/Host execution target. */
    XKRT_TASK_FORMAT_TARGET_CUDA  = 1, /**< NVIDIA CUDA execution target. */
    XKRT_TASK_FORMAT_TARGET_ZE    = 2, /**< Intel Level Zero (oneAPI) execution target. */
    XKRT_TASK_FORMAT_TARGET_CL    = 3, /**< OpenCL execution target. */
    XKRT_TASK_FORMAT_TARGET_HIP   = 4, /**< AMD HIP execution target. */
    XKRT_TASK_FORMAT_TARGET_SYCL  = 5, /**< SYCL execution target. */
    XKRT_TASK_FORMAT_TARGET_MAX   = 6  /**< Marker for the total number of targets. */

}               xkrt_task_format_target_t;

/**
 * @def XKRT_TASK_FORMAT_TARGET_NO_SUGGEST
 * @brief Indicates that no specific target is suggested.
 *
 * This macro uses the `XKRT_TASK_FORMAT_TARGET_MAX` value as a sentinel
 * to signify that the suggestion function does not have a preference
 * for the execution target.
 */
# define XKRT_TASK_FORMAT_TARGET_NO_SUGGEST XKRT_TASK_FORMAT_TARGET_MAX

/**
 * @typedef xkrt_task_format_suggest_t
 * @brief Function pointer type for a task target suggestion function.
 *
 * A function of this type can be used to dynamically suggest the best
 * execution target for a given task.
 *
 * @param task A pointer to the task data.
 * @return The suggested ::xkrt_task_format_target_t.
 */
typedef xkrt_task_format_target_t (*xkrt_task_format_suggest_t)(void * task);

/**
 * @typedef xkrt_task_format_func_t
 * @brief Function pointer type for a task's implementation.
 *
 * This defines the signature for the actual function that executes a task.
 * It takes no arguments and returns void. Task-specific data is expected
 * to be managed through other mechanisms (e.g., captured in a lambda,
 * passed via a task structure).
 */
typedef void (*xkrt_task_format_func_t)();

/**
 * @struct xkrt_task_format_t
 * @brief Holds the implementations for a single task format across different targets.
 *
 * This structure aggregates the function pointers for a specific task's
 * implementation on all available execution targets. It also includes
 * a label for identification and an optional suggestion function.
 */
typedef struct  xkrt_task_format_t
{
    /**
     * @brief Array of function pointers for each target.
     *
     * `f[target]` holds the function to execute for the corresponding
     * ::xkrt_task_format_target_t.
     */
    xkrt_task_format_func_t f[XKRT_TASK_FORMAT_TARGET_MAX];

    /**
     * @brief Optional source code attached to each target's function.
     *
     * `source[target]` may hold the program source (e.g. LLVM-IR) for the
     * function `f[target]`. It follows the same representation as CGIR's
     * `command_prog_t.source` so that, when a task dependency graph is lowered
     * to a CGIR command graph, the source can be forwarded to the generated
     * `COMMAND_TYPE_PROG` commands (enabling CGIR optimization passes such as
     * program/loop fusion). A zero-initialized entry (`content.llvmir.raw ==
     * NULL`) means "no source".
     */
    cgir_command_prog_source_t source[XKRT_TASK_FORMAT_TARGET_MAX];

    /**
     * @brief A human-readable label for the task format.
     */
    char label[32];

    /**
     * @brief Optional function to suggest an execution target.
     *
     * If non-NULL, this function is called to dynamically determine
     * the best target for a task.
     * @param task A pointer to the task data.
     * @return The suggested ::xkrt_task_format_target_t, or
     * ::XKRT_TASK_FORMAT_TARGET_NO_SUGGEST if no preference.
     */
    xkrt_task_format_target_t (*suggest)(void * task);

} xkrt_task_format_t;

/**
 * @typedef xkrt_task_format_id_t
 * @brief A unique identifier for a task format.
 *
 * This type is used as an index into the ::xkrt_task_formats_t list.
 *
 * It is 16-bit wide to allow runtimes (e.g. XKOMP) to register one task format
 * per source location without quickly exhausting the id space.
 */
typedef uint16_t xkrt_task_format_id_t;

/**
 * @def XKRT_TASK_FORMAT_MAX
 * @brief The maximum number of task formats that can be registered.
 *
 * This is decoupled from the (wider) ::xkrt_task_format_id_t range so that the
 * statically-sized ::xkrt_task_formats_t::list array stays reasonable in size.
 * It can be overridden at build time.
 */
# ifndef XKRT_TASK_FORMAT_MAX
#  define XKRT_TASK_FORMAT_MAX 4096
# endif
# ifdef __cplusplus
static_assert(XKRT_TASK_FORMAT_MAX <= ((1 << (sizeof(xkrt_task_format_id_t) * 8)) - 1),
        "XKRT_TASK_FORMAT_MAX must fit in xkrt_task_format_id_t");
# endif

/**
 * @def XKRT_TASK_FORMAT_NULL
 * @brief Represents a null or invalid task format ID.
 */
# define XKRT_TASK_FORMAT_NULL 0

/**
 * @struct xkrt_task_formats_t
 * @brief A repository for all registered task formats.
 *
 * This structure manages a list of all available task formats,
 * allowing them to be accessed by their ::xkrt_task_format_id_t.
 */
typedef struct  xkrt_task_formats_t
{
    /**
     * @brief Array holding all registered task formats.
     */
    xkrt_task_format_t list[XKRT_TASK_FORMAT_MAX];

    /**
     * @brief The ID to be assigned to the next registered task format.
     *
     * This is typically managed atomically.
     */
    xkrt_task_format_id_t next_fmtid;
}               xkrt_task_formats_t;

#endif /* __XKRT_TASK_FORMAT_H__ */
