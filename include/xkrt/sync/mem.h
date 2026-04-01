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

#ifndef __MEM_H__
# define __MEM_H__

/**
 *  Retrieved from kaapi_atomic.h
 */

#if defined(__i386__)|| defined(__x86_64)
#  if !defined(__MIC__)
#      define mem_pause() do { __asm__ __volatile__("pause\n\t"); } while (0)
#  else
#    define mem_pause() do { __asm__ __volatile__("mov $20, %rax; delay %rax\n\t"); __asm__ __volatile__ ("":::"memory"); } while (0)
#  endif
#else
#  define mem_pause()
#endif

#if defined(__APPLE__)
#  include <libkern/OSAtomic.h>
static inline void writemem_barrier()
{
#  if defined(__x86_64) || defined(__i386__)
  /* not need sfence on X86 archi: write are ordered __asm__ __volatile__ ("sfence":::"memory"); */
  __asm__ __volatile__ ("":::"memory");
#  else
  OSMemoryBarrier();
#  endif
}

static inline void readmem_barrier()
{
#  if defined(__x86_64) || defined(__i386__)
  __asm__ __volatile__ ("":::"memory");
//  __asm__ __volatile__ ("lfence":::"memory");
#  else
  OSMemoryBarrier();
#  endif
}

/* should be both read & write barrier */
static inline void mem_barrier()
{
#  if defined(__x86_64) || defined(__i386__)
  /** Mac OS 10.6.8 with gcc 4.2.1 has a buggy __sync_synchronize();
      gcc-4.4.4 pass the test with sync_synchronize
  */
#if !defined(__MIC__)
  __asm__ __volatile__ ("mfence":::"memory");
#endif
#  else
  OSMemoryBarrier();
#  endif
}

#elif defined(__linux__)    /* defined(__APPLE__) */
static inline void writemem_barrier()
{
#  if defined(__x86_64) || defined(__i386__)
  /* not need sfence on X86 archi: write are ordered */
  __asm__ __volatile__ ("":::"memory");
#  elif defined(__GNUC__)
  __sync_synchronize();
#  else
#  error "Compiler not supported"
/* xlC ->__lwsync() / bultin */
#  endif
}

static inline void readmem_barrier()
{
#  if defined(__x86_64) || defined(__i386__)
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("":::"memory");
#  elif defined(__GNUC__)
  __sync_synchronize();
#  else
#  error "Compiler not supported"
/* xlC ->__lwsync() / bultin */
#  endif
}

/* should be both read & write barrier */
static inline void mem_barrier()
{
#  if defined(__GNUC__) || defined(__ICC)
  __sync_synchronize();
#  elif defined(__x86_64) || defined(__i386__)
  __asm__ __volatile__ ("mfence":::"memory");
#elif defined(__arm__) || defined(__aarch64__)
  __asm__ __volatile__ ("dmb ish":::"memory");
#  else
#   error "Compiler not supported"
/* bultin ?? */
#  endif
}

#elif defined(_WIN32)
static inline void writemem_barrier()
{
  /* Compiler fence to keep operations from */
  /* not need sfence on X86 archi: write are ordered */
  __asm__ __volatile__ ("":::"memory");
}

static inline void readmem_barrier()
{
  /* Compiler fence to keep operations from */
  /* not need lfence on X86 archi: read are ordered */
  __asm__ __volatile__ ("":::"memory");
}

#else
#  error "Undefined barrier"
#endif

#endif /* __MEM_H__ */
