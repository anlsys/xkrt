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

#ifndef __SMALL_VECTOR_H__
# define __SMALL_VECTOR_H__

# include <assert.h>
# include <stdlib.h>
# include <string.h>
# include <span>

// Made with the assistance of Clause Sonnet 4.6

// ---------------------------------------------------------------------------
// small_vector_t<T, N>: inline storage for N elements, heap overflow beyond that.
// Zero heap allocation for the common case
// ---------------------------------------------------------------------------
template <typename T, int N = 8>
class small_vector_t
{
    T    _buf[N];
    int  _sz           = 0;
    T *  _overflow     = nullptr;
    int  _overflow_cap = 0;

    inline T * data_ptr()             { return _overflow_cap ? _overflow : _buf; }
    inline const T * data_ptr() const { return _overflow_cap ? _overflow : _buf; }

    void grow() {
        int nc = _overflow_cap ? _overflow_cap * 2 : N * 2;
        T * nb = static_cast<T *>(realloc(_overflow, nc * sizeof(T)));
        assert(nb);
        if (!_overflow_cap)
            memcpy(nb, _buf, _sz * sizeof(T));
        _overflow     = nb;
        _overflow_cap = nc;
    }

public:
    small_vector_t()  = default;
    ~small_vector_t() { if (_overflow) free(_overflow); }

    small_vector_t(const small_vector_t &)            = delete;
    small_vector_t & operator=(const small_vector_t&) = delete;

    small_vector_t(small_vector_t && o) noexcept
        : _sz(o._sz), _overflow(o._overflow), _overflow_cap(o._overflow_cap)
    {
        if (!_overflow_cap)
            memcpy(_buf, o._buf, _sz * sizeof(T));
        o._overflow = nullptr; o._overflow_cap = 0; o._sz = 0;
    }

    small_vector_t & operator=(small_vector_t && o) noexcept {
        if (this != &o) {
            if (_overflow) free(_overflow);
            _sz = o._sz; _overflow = o._overflow; _overflow_cap = o._overflow_cap;
            if (!_overflow_cap) memcpy(_buf, o._buf, _sz * sizeof(T));
            o._overflow = nullptr; o._overflow_cap = 0; o._sz = 0;
        }
        return *this;
    }

    inline int  size()  const { return _sz; }
    inline bool empty() const { return _sz == 0; }

    inline void
    push_back(T v)
    {
        if (__builtin_expect(
                (!_overflow_cap && _sz == N) ||
                ( _overflow_cap && _sz == _overflow_cap), 0))
            grow();
        data_ptr()[_sz++] = v;
    }

    inline T &
    back(void)
    {
        return data_ptr()[_sz - 1];
    }

    inline const T &
    back(void) const
    {
        return data_ptr()[_sz - 1];
    }

    // O(1) erase — order not preserved, fine for predecessor sets
    inline void
    swap_erase(int i)
    {
        data_ptr()[i] = data_ptr()[--_sz];
    }

    inline void
    clear(void)
    {
        _sz = 0;
    }

    inline T *
    emplace_back()
    {
        if (__builtin_expect(
                (!_overflow_cap && _sz == N) ||
                ( _overflow_cap && _sz == _overflow_cap), 0))
            grow();
        return &data_ptr()[_sz++];
    }

    inline void
    pop_back()
    {
        --_sz;
    }

    inline T & operator[](int i)       { return data_ptr()[i]; }
    inline T   operator[](int i) const { return data_ptr()[i]; }

    inline T *       begin()       { return data_ptr(); }
    inline T *       end()         { return data_ptr() + _sz; }
    inline const T * begin() const { return data_ptr(); }
    inline const T * end()   const { return data_ptr() + _sz; }

    inline std::span<T>       span()       { return { data_ptr(), static_cast<size_t>(_sz) }; }
    inline std::span<const T> span() const { return { data_ptr(), static_cast<size_t>(_sz) }; }
};

#endif /* __SMALL_VECTOR_H__ */
