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

# include <random>

# include <xkrt/runtime.h>
# include <xkrt/logger/logger.h>
# include <xkrt/logger/metric.h>

XKRT_NAMESPACE_USE;

int
main(int argc, char ** argv)
{
    runtime_t runtime;

    if (runtime.init())
        LOGGER_FATAL("ERROR INIT");

    driver_t * driver = runtime.driver_get(XKRT_DRIVER_TYPE_HOST);
    assert(driver);

    std::mt19937 rng(std::random_device{}());
    int done[3] = {0, 0, 0};

    if (argc == 2)
    {
        done[0] = 1;
        done[1] = 1;
        done[2] = 1;
        done[atoi(argv[1])] = 0;
    }

    int dumped = 0;

    while (done[0] + done[1] + done[2] < 3)
    {
        int i;
        do { i = rng() % 3; } while (done[i]);
        done[i] = 1;

        LOGGER_INFO("------------------------------");

        # include "memory-register-async.conf.cc"
        if (dumped == 0)
        {
            LOGGER_INFO("Size is %.1f GB in %zu chunks using %u threads",
                size/1e9, nchunks, team->priv.nthreads);
            dumped = 1;
        }

        uint64_t t0 = get_nanotime();

        if (i == 0 || i == 1)
        {
            if (i == 0)
            {
                LOGGER_INFO("Running with touch>sync>register>sync>unregister>sync");

                uint64_t t0 = get_nanotime();
                runtime.memory_touch_async(team, ptr, size, nchunks);
                runtime.task_wait();
                uint64_t tf = get_nanotime();
                LOGGER_INFO("      Touch took %lf s.", (tf - t0) / 1e9);
            }
            else if (i == 1)
            {
                LOGGER_INFO("Running with register>sync>unregister>sync");
            }

            {
                uint64_t t0 = get_nanotime();
                runtime.memory_register_async(team, ptr, size, nchunks);
                runtime.task_wait();
                uint64_t tf = get_nanotime();
                LOGGER_INFO("    Pinning took %lf s.", (tf - t0) / 1e9);
            }
        }
        else if (i == 2)
        {
            LOGGER_INFO("Running with touch>register>sync>unregister>sync");
            uint64_t t0 = get_nanotime();
            runtime.memory_touch_async(team, ptr, size, nchunks);
            runtime.memory_register_async(team, ptr, size, nchunks);
            runtime.task_wait();
            uint64_t tf = get_nanotime();
            LOGGER_INFO("  Touch+Pin took %lf s.", (tf - t0) / 1e9);
        }

        {
            uint64_t t0 = get_nanotime();
            runtime.memory_unregister_async(team, ptr, size, nchunks);
            runtime.task_wait();
            uint64_t tf = get_nanotime();
            LOGGER_INFO("  Unpinning took %lf s.", (tf - t0) / 1e9);
        }

        uint64_t tf = get_nanotime();
        LOGGER_INFO("Total took %lf s.", (tf - t0) / 1e9);
    }

    if (runtime.deinit())
        LOGGER_FATAL("ERROR DEINIT");

    return 0;
}
