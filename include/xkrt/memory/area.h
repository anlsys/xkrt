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

#ifndef __XKRT_AREA_H__
# define __XKRT_AREA_H__

# include <xkrt/sync/mutex.h>
# include <stdint.h>

///////////////////////////
// Driver devices memory //
///////////////////////////

typedef enum    xkrt_area_chunk_state_t
{
    XKRT_ALLOC_CHUNK_STATE_FREE       = 0,
    XKRT_ALLOC_CHUNK_STATE_ALLOCATED  = 1,

}               xkrt_area_chunk_state_t;

/**
 * Represent a segment of memory in device memory (used by custom allocator)
 * It is placed in two chained list:
 *  - the list of all chunk in device memory
 *  - the list of free chunk in device memory
*/
typedef struct  xkrt_area_chunk_t
{
    uintptr_t ptr;                          /* position of memory in device */
    size_t size;                            /* size of the segment in byte */
    int state;                              /* state of the chunk */
    struct xkrt_area_chunk_t * prev;        /* previous chunk in double chained list */
    struct xkrt_area_chunk_t * next;        /* next chunk in double chained list */
    struct xkrt_area_chunk_t * freelink;    /* next freechunk in the chained list */
    int use_counter;                        /* used in the memory-tree to count how many blocks relies on that allocation chunk */
    int area_idx;                           /* memory area index in the device (TODO: bad design) */
}               xkrt_area_chunk_t;

/* The device memory with allocation information */
typedef struct  xkrt_area_t
{
    xkrt_mutex_t lock;
    xkrt_area_chunk_t * free_chunk_list;

}               xkrt_area_t;

#endif /* __XKRT_AREA_H__ */
