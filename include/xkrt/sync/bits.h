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
#ifndef __BITS_H__
# define __BITS_H__

# include <xkrt/consts.h>

XKRT_NAMESPACE_BEGIN

/* utility: return the index+1 of random bit set to '1' */
static inline device_unique_id_t
__random_set_bit(device_unique_id_bitfield_t bitfield)
{
    static thread_local unsigned int seed = 0x42;

    if (bitfield == 0)
        LOGGER_FATAL("Tried to get a random bit from a NULL bitfield");

    /* must be true, as 'builtin_popcount' works on 'int' type */
    static_assert(sizeof(device_unique_id_bitfield_t) <= sizeof(int));

    const int nb = __builtin_popcount(bitfield);
    device_unique_id_t idx = 0;
    int k = rand_r(&seed) % nb;
    for (int i = 0; i <= k; ++i)
    {
        idx = static_cast<device_unique_id_t>(__builtin_ffs(static_cast<int>(bitfield)));
        bitfield &= ~(1u << (idx - 1));
    }

    return idx;
}

XKRT_NAMESPACE_END

#endif /* __BITS_H__ */
