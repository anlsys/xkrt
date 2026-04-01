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
 *  Idea is to use a dijkstra algorithm with weights being BW, but ignoring the link if there is already a pending transfer
 *  Problem is it may add significant latency
 */

# ifndef __ROUTER_CFS_HPP__
#  define __ROUTER_CFS_HPP__

# include <xkrt/consts.h>
# include <xkrt/memory/routing/router.hpp>
# include <xkrt/sync/bits.h>

# include <stdint.h>

/** The higher the rank, the lower the performance */
class RouterCFS : public Router
{
    public:
        typedef struct xkrt_router_cfs_map_t
        {
            struct {
                uint8_t weight;
                bool used;
            } values[XKRT_DEVICES_MAX][XKRT_DEVICES_MAX];

            xkrt_router_cfs_map_t(const uint8_t weights[XKRT_DEVICES_MAX][XKRT_DEVICES_MAX])
            {
                for (xkrt_device_unique_id_t i = 0 ; i < XKRT_DEVICES_MAX ; ++i)
                {
                    for (xkrt_device_unique_id_t j = 0 ; j < XKRT_DEVICES_MAX ; ++j)
                    {
                        values[i][j].weight = weights[i][j];
                        values[i][j].used = false;
                    }
                }
            }
        }               xkrt_router_cfs_map_t;

    public:

        /**
         *  A graph where weights[i][j] is the weight of the edge between node 'i' and 'j'
         *  If weights[i][j] is UINT8_MAX, then there is no edge
         *  If weights[i][j] = weights[i'][j'] / x then the bw on link (i,j) is
         *      'x' times greater than the bw of link (i,j) - i.e. the lower weights[i][j] the greater the BW
         *  If used[i][j], then a communication is already occuring on link (i,j)
         */
        const xkrt_router_cfs_map_t map;

    public:

        RouterCFS(const uint8_t weights[XKRT_DEVICES_MAX][XKRT_DEVICES_MAX]) : weights(weights) {}
        ~RouterCFS() {}

        /* @override */
        xkrt_device_unique_id_t
        get_source(
            const xkrt_device_unique_id_t dst,
            const xkrt_device_unique_id_bitfield_t valid
        ) const {

            /* fast way out: valid on that device already */
            if (valid & (1 << dst))
                return dst;

            // TODO

            /* get any random device */
            return (xkrt_device_unique_id_t) (__random_set_bit(valid) - 1);
        }

};

# endif /* __ROUTER_CFS_HPP__ */
