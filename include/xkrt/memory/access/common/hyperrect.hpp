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

#ifndef __HYPERCUBE_HPP__
# define __HYPERCUBE_HPP__

# include <xkrt/utils/min-max.h>
# include <xkrt/memory/access/common/interval.hpp>

# include <cassert>
# include <cstdlib>
# include <ostream>
# include <iostream>
# include <cstring>

/* K is the number of dimensions */
template<int K>
struct KHyperrect
{
    Interval list[K];

    Interval   operator [](int i) const { return this->list[i]; }
    Interval & operator [](int i) { return this->list[i]; }

    void
    set_list(const Interval src[K])
    {
        memcpy(this->list, src, sizeof(this->list));
    }

    void
    tostring(char * buffer, int size) const
    {
        int r = 0;
        for (int k = 0 ; k < K ; ++k)
        {
            r += snprintf(buffer + r, size - r, "[" INTERVAL_TYPE_MODIFIER ".." INTERVAL_TYPE_MODIFIER "[%c",
                    this->list[k].a, this->list[k].b,
                    (k == K - 1) ? '\0' : '\n');
            if (r == size)
                break ;
        }
    }

    // return true if intervals intersects on each dimension
    inline bool
    intersects(const KHyperrect & intervals) const
    {
        for (int k = 0 ; k < K ; ++k)
        {
            if (this->list[k].a < intervals.list[k].b && this->list[k].b > intervals.list[k].a)
                continue ;
            return false;
        }
        return true;
    }

    static inline void
    intersection(
        KHyperrect * dst,
        const KHyperrect & x,
        const KHyperrect & y
    ) {
        for (int k = 0 ; k < K ; ++k)
        {
            dst->list[k].a = MAX(x.list[k].a, y.list[k].a);
            dst->list[k].b = MIN(x.list[k].b, y.list[k].b);
        }
    }

    inline bool
    intersects(const KHyperrect * & intervals) const
    {
        return this->intersects(*intervals);
    }

    inline bool
    equals(const KHyperrect & intervals) const
    {
        for (int k = 0 ; k < K ; ++k)
        {
            if (this->list[k] == intervals.list[k])
                continue ;
            return false;
        }
        return true;
    }

    inline bool
    includes(const KHyperrect & intervals, int k) const
    {
        for ( ; k < K ; ++k)
        {
            if (this->list[k].includes(intervals.list[k]))
                continue ;
            return false;
        }
        return true;
    }

    inline bool
    includes(const KHyperrect & intervals) const
    {
        return this->includes(intervals, 0);
    }

    inline bool
    is_empty(void) const
    {
        for (int i = 0 ; i < K ; ++i)
            if (this->list[i].a >= this->list[i].b)
                return true;
        return false;
    }

    inline uint64_t
    size(void) const
    {
        uint64_t s = this->list[0].length();
        for (int i = 1 ; i < K ; ++i)
            s *= this->list[i].length();
        return s;
    }

    /* return the distance | y - x | between 'left-top' corners of each
     * dimensions of the interval */
    static inline void
    distance_manhattan(
        const KHyperrect & x,
        const KHyperrect & y,
        INTERVAL_DIFF_TYPE_T d[K]
    ) {
        for (int k = 0 ; k < K ; ++k)
            d[k] = (INTERVAL_DIFF_TYPE_T) (y.list[k].a - x.list[k].a);
    }

    friend std::ostream &
    operator<<(std::ostream & os, const KHyperrect & intervals)
    {
        for (int k = 0 ; k < K ; ++k)
        {
            os << "(" << intervals[k].a << "," << intervals[k].b << ")";
            if (k != K-1)
                os << "x";
        }
        return os;
    }

}; /* class KHyperrect */

using Hyperrect = KHyperrect<2>;

#endif /* __HYPERCUBE_HPP__ */
