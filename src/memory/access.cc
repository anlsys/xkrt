/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
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

# include <xkrt/memory/access/access.hpp>
# include <xkrt/logger/logger.h>

XKRT_NAMESPACE_BEGIN;

bool
access_t::intersects(
    access_t * x,
    access_t * y
) {
    assert(x->type == y->type);
    switch (x->type)
    {
        case (ACCESS_TYPE_HANDLE):
            return x->region.point.handle == y->region.point.handle;

        case (ACCESS_TYPE_SEGMENT):
            return x->region.interval.segment.intersects(y->region.interval.segment);

        case (ACCESS_TYPE_BLAS_MATRIX):
        {
            if (x->host_view.ld * x->host_view.sizeof_type == x->host_view.ld * x->host_view.sizeof_type)
            {
                for (Rect & rx : x->region.matrix.rects)
                    for (Rect & ry : y->region.matrix.rects)
                        if (rx.intersects(ry))
                            return true;
                return false;
            }
            else
            {
                LOGGER_FATAL("Not supported");
            }
            break ;
        }

        default:
            LOGGER_FATAL("Not implemented");
            return false;
    }
}

bool
access_t::conflicts(
    access_t * x,
    access_t * y
) {
    if (!(x->mode & ACCESS_MODE_W) & !(y->mode & ACCESS_MODE_W))
        return false;
    return access_t::intersects(x, y);
}

void
access_t::split(
    access_t * x,
    access_t * y,
    task_t * y_task,
    access_split_mode_t split_mode
) {
    switch (x->type)
    {
        case (ACCESS_TYPE_SEGMENT):
        {
            //        a                 b
            //  x = [ . . . . . . . . . [
            //  then
            //        a      a+h
            //  y = [ . . . . [         b
            //  x =           [ . . . . [

            const uintptr_t a = x->region.interval.segment[0].a;
            const uintptr_t b = x->region.interval.segment[0].b;
            const uintptr_t h = (b - a) / 2;
            new (y) access_t( y_task, a + 0, a + h, x->mode, x->concurrency, x->scope);
            new (x) access_t(x->task, a + h, b + 0, x->mode, x->concurrency, x->scope);

            break ;
        }

        // TODO
        case (ACCESS_TYPE_BLAS_MATRIX):
        {
            switch (split_mode)
            {
                //  x x x x    y1 y1 y2 y2
                //  x x x x -> y1 y1 y2 y2
                //  x x x x    y3 y3 x  x
                //  x x x x    y3 y3 x  x
                case (ACCESS_SPLIT_MODE_QUADRANT):
                {
                    LOGGER_FATAL("IMPL ME");
                    break ;
                }

                //  x x x x    y y y y
                //  x x x x -> y y y y
                //  x x x x    x x x x
                //  x x x x    x x x x
                case (ACCESS_SPLIT_MODE_HALVES):
                case (ACCESS_SPLIT_MODE_HALVES_HORIZONTAL):
                {
                    LOGGER_FATAL("IMPL ME");
                    break ;
                }

                //  x x x x    y y x x
                //  x x x x -> y y x x
                //  x x x x    y y x x
                //  x x x x    y y x x
                case (ACCESS_SPLIT_MODE_HALVES_VERTICAL):
                {
                    LOGGER_FATAL("IMPL ME");
                    break ;
                }

                case (ACCESS_SPLIT_MODE_NO_SPLIT):
                case (ACCESS_SPLIT_MODE_CUSTOM):
                {
                    LOGGER_FATAL("Not supported");
                    break ;
                }
            }

            break ;
        }

        case (ACCESS_TYPE_HANDLE):
        default:
        {
            // TODO: this should be provided by the user somehow
            LOGGER_FATAL("Not implemented");
            break ;
        }
    }
}

XKRT_NAMESPACE_END
