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

#ifndef __LOGGER_HIPBLAS_H__
# define __LOGGER_HIPBLAS_H__

# include <xkrt/logger/logger.h>
# include <hipblas/hipblas.h>

static const char *
hipblas_error_to_str(hipblasStatus_t status)
{
    # if 0
    HIPBLAS_STATUS_SUCCESS                // Operation completed successfully
    HIPBLAS_STATUS_NOT_INITIALIZED  = 1,  // The library was not initialized
    HIPBLAS_STATUS_ALLOC_FAILED     = 2,  // Resource allocation failed
    HIPBLAS_STATUS_INVALID_VALUE    = 3,  // Invalid value was passed
    HIPBLAS_STATUS_MAPPING_ERROR    = 4,  // Memory mapping error
    HIPBLAS_STATUS_EXECUTION_FAILED = 5,  // Execution of GPU operations failed
    HIPBLAS_STATUS_INTERNAL_ERROR   = 6,  // An internal library error occurred
    HIPBLAS_STATUS_NOT_SUPPORTED    = 7,  // The operation is not supported
    HIPBLAS_STATUS_ARCH_MISMATCH    = 8   // The device architecture is not supported
    # endif

    static const char * names[] = {
        "SUCCESS",
        "NOT_INITIALIZED",
        "ALLOC_FAILED",
        "INVALID_VALUE",
        "MAPPING_ERROR",
        "EXECUTION_FAILED",
        "INTERNAL_ERROR",
        "NOT_SUPPORTED",
        "ARCH_MISMATCH"
    };
    const char * name  = (status >= 0 && status <= 8) ? names[status] : NULL;
    return name;
}

# define HIPBLAS_SAFE_CALL(X)                                                               \
    do {                                                                                    \
        hipblasStatus_t r = X;                                                              \
        if (r != HIPBLAS_STATUS_SUCCESS)                                                    \
            LOGGER_FATAL("`%s` failed with err=%s (%d)", #X, hipblas_error_to_str(r), r);   \
    } while (0)

#endif /* __LOGGER_HIPBLAS_H__ */
