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

# ifndef __ROUTER_RANDOM_HPP__
#  define __ROUTER_RANDOM_HPP__

# include <xkrt/memory/routing/router.hpp>
# include <xkrt/sync/bits.h>

/** The higher the rank, the lower the performance */
class RouterRandom : public Router
{
    # if 0
    private:
        /** The lower the rank, the higher the performance */
        const xkrt_device_unique_id_bitfield_t affinity[XKRT_DEVICES_PERF_RANK_MAX];
    # endif /* 0 */

    public:
        RouterRandom() {}
        ~RouterRandom() {}

        xkrt_device_unique_id_t
        get_source(
            const xkrt_device_unique_id_t dst,
            const xkrt_device_unique_id_bitfield_t valid
        ) const {
            /* get any random device */
            return (xkrt_device_unique_id_t) (__random_set_bit(valid) - 1);
        }

};

# endif /* __ROUTER_RANDOM_HPP__ */
