/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Thierry Gautier, thierry.gautier@inrialpes.fr
** Joao Lima joao.lima@inf.ufsm.br
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

# include <xkrt/runtime.h>
# include <xkrt/memory/access/blas/dependency-tree.hpp>

# include <math.h>

XKRT_NAMESPACE_USE;

//////////////////
// DISTRIBUTION //
//////////////////

extern "C"
void
distribution1D_init(
    distribution_t * d,
    distribution_type_t type,
    size_t count,
    size_t size, size_t bs
) {
    assert(count);

    d->type  = type;
    d->count = count;
    d->size  = size;
    d->bs    = bs;
    d->t     = NUM_OF_TILES(size, bs);

    switch (type)
    {
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC1D):
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC2D):
        {
            // nothing to do
            break ;
        }

        case (XKRT_DISTRIBUTION_TYPE_CYCLIC2DBLOCK):
        {

            /* find the most square decomposition of count in d->gm x d->gn */
            d->blkm = 1;
            d->blkn = 1;
            d->gm = (size_t) sqrt((double) count);
            d->gn = d->gm;
            if (d->gm == 0)
            {
                d->gm = 1;
                d->gn = count;
            }
            else
            {
                size_t g;
                for (g = d->gm + 1; g > 0; --g)
                    if (count % g == 0)
                        break;

                # pragma message(TODO "Why this inverts with the previous case ?")
                if (g == 0)
                {
                    d->gm = count; // = 1
                    d->gn = 1;     // = count
                }
                else
                {
                    d->gm = g;
                    d->gn = count / g;
                }
            }

            d->blkm = d->blkm;
            d->blkn = d->blkn;
            d->gm   = d->gm;
            d->gn   = d->gn;

            break ;
        }

        default:
            LOGGER_FATAL("Not implemented");
    }
}

extern "C"
void
distribution2D_init(
    distribution_t * d,
    distribution_type_t type,
    size_t count,
    size_t m, size_t n,
    size_t mb, size_t nb
) {
    assert(count);

    d->type  = type;
    d->count = count;
    d->m     = m;
    d->n     = n;
    d->mb    = mb;
    d->nb    = nb;
    d->mt    = NUM_OF_TILES(m, mb);
    d->nt    = NUM_OF_TILES(n, nb);

    switch (type)
    {
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC1D):
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC2D):
        {
            // nothing to do
            break ;
        }

        case (XKRT_DISTRIBUTION_TYPE_CYCLIC2DBLOCK):
        {

            /* find the most square decomposition of count in d->gm x d->gn */
            d->blkm = 1;
            d->blkn = 1;
            d->gm = (size_t) sqrt((double) count);
            d->gn = d->gm;
            if (d->gm == 0)
            {
                d->gm = 1;
                d->gn = count;
            }
            else
            {
                size_t g;
                for (g = d->gm + 1; g > 0; --g)
                    if (count % g == 0)
                        break;

                # pragma message(TODO "Why this inverts with the previous case ?")
                if (g == 0)
                {
                    d->gm = count; // = 1
                    d->gn = 1;     // = count
                }
                else
                {
                    d->gm = g;
                    d->gn = count / g;
                }
            }

            d->blkm = d->blkm;
            d->blkn = d->blkn;
            d->gm   = d->gm;
            d->gn   = d->gn;

            break ;
        }

        default:
            LOGGER_FATAL("Not implemented");
    }
}

extern "C"
device_unique_id_t
distribution1D_get(
    distribution_t * d,
    size_t t
) {
    assert(t < d->t);

    switch (d->type)
    {
        /** example of 4 gpus
         *  1 2 3 4 1 2 3
         */
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC1D):
            return 1 + (device_unique_id_t) (t % d->count);

        default:
            LOGGER_FATAL("Not implemented");
            return 0;
    }
}

extern "C"
device_unique_id_t
distribution2D_get(
    distribution_t * d,
    size_t tm, size_t tn
) {
    assert(tm < d->mt);
    assert(tn < d->nt);

    switch (d->type)
    {
        /**
         * example on 4 gpus
         *  1 2 3 4
         *  1 2 3 4
         *  1 2 3 4
         *  1 2 3 4
         */
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC2D):
            return 1 + (device_unique_id_t) ((tm * d->nt + tn) % d->count);

        /**
         * example on 4 gpus
         *  1 2 1 2
         *  3 4 3 4
         *  1 2 1 2
         *  3 4 3 4
         */
        case (XKRT_DISTRIBUTION_TYPE_CYCLIC2DBLOCK):
            return 1 + (device_unique_id_t) (
                    (
                     ((tm / d->blkm) % d->gm) * d->gn +
                      (tn / d->blkn) % d->gn)
                        % d->count
                    )
                ;
        default:
            LOGGER_FATAL("Not implemented");
            return 0;
    }
}
