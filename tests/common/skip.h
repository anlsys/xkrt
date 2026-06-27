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
 *  Shared test helpers for the XKRT test-suite.
 *
 *  These helpers do NOT replace `assert(...)`: tests keep using plain
 *  `assert` for correctness checks (the test targets are always compiled
 *  with assertions enabled, see tests/CMakeLists.txt).
 *
 *  The only purpose of this header is to let a test that *requires* a GPU
 *  device gracefully *skip* itself (instead of aborting) when run on a
 *  machine that has no GPU. A skipped test:
 *      - emits a warning explaining why it was ignored, and
 *      - returns `XKRT_TEST_SKIP_RETURN_CODE`.
 *
 *  CTest is configured (via the `SKIP_RETURN_CODE` test property, set by
 *  `xkrt_add_test(... GPU)`) to report such a return value as "Skipped"
 *  rather than "Passed" or "Failed".
 */

#ifndef __XKRT_TEST_SKIP_H__
# define __XKRT_TEST_SKIP_H__

# include <xkrt/logger/logger.h>
# include <xkrt/runtime.h>

/* Return code that CTest interprets as "this test was skipped".
 * Keep in sync with `SKIP_RETURN_CODE` in tests/CMakeLists.txt.
 * 77 is the historical autotools convention for a skipped test. */
# define XKRT_TEST_SKIP_RETURN_CODE 77

/**
 *  Skip the calling `int main(...)` (returning XKRT_TEST_SKIP_RETURN_CODE)
 *  when the runtime sees no GPU device. The host is always device 0, so a
 *  GPU is present only when `get_ndevices() > 1`.
 *
 *  `rt` must be an already-initialized lvalue of type `xkrt::runtime_t`.
 *  The runtime is deinitialized before returning so resources are released.
 */
# define XKRT_TEST_SKIP_IF_NO_GPU(rt)                                           \
    do {                                                                        \
        if ((rt).get_ndevices() <= 1)                                          \
        {                                                                       \
            LOGGER_WARN(                                                        \
                "SKIP: this test requires a GPU device but none is available " \
                "(get_ndevices() = %u, only the host is present). Ignoring.",  \
                (rt).get_ndevices());                                          \
            (rt).deinit();                                                      \
            return XKRT_TEST_SKIP_RETURN_CODE;                                  \
        }                                                                       \
    } while (0)

#endif /* __XKRT_TEST_SKIP_H__ */
