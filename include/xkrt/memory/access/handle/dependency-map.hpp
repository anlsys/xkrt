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

// OpenMP 6.0-alike dependencies

#ifndef __DEPENDENCY_MAP_HPP__
# define __DEPENDENCY_MAP_HPP__

# include <xkrt/data-structures/small-vector.h>
# include <xkrt/memory/access/dependency-domain.hpp>
# include <xkrt/runtime.h>
# include <xkrt/support.h>
# include <xkrt/task/task.hpp>

# include <cstring>
# include <cassert>
# include <utility>

XKRT_NAMESPACE_BEGIN

// Clause Sonnet 4.6 assisted

// ---------------------------------------------------------------------------
// FlatPtrMap<V>: open-addressing hash map with inline value storage.
//
// - One contiguous allocation: no per-entry heap node, no pointer chasing.
// - Hash can be configured to
//      - Knuth multiplicative hash on pointer keys (ptr >> 3 strips alignment).
//      - Jenkins hash (better occupancy) <- default one
// - Load factor kept <= 0.5 for O(1) average probe length.
// - V must be default-constructible and move-constructible.
//
// WARNING: try_emplace() may rehash, which invalidates ALL live V pointers.
//          Callers must re-fetch via find() after any try_emplace() call.
//          find() never rehashes and is always safe.
// ---------------------------------------------------------------------------
template <typename V>
class FlatPtrMap
{
    struct Slot {
        const void * key      = nullptr;
        bool         occupied = false;
        alignas(V) char val[sizeof(V)];
    };

    Slot * _slots = nullptr;
    int    _cap   = 0;
    int    _sz    = 0;

    // From UTHASH library
    # define HASH_JEN_MIX(a,b,c)                                                     \
    do {                                                                             \
      a -= b; a -= c; a ^= ( c >> 13 );                                              \
      b -= c; b -= a; b ^= ( a << 8 );                                               \
      c -= a; c -= b; c ^= ( b >> 13 );                                              \
      a -= b; a -= c; a ^= ( c >> 12 );                                              \
      b -= c; b -= a; b ^= ( a << 16 );                                              \
      c -= a; c -= b; c ^= ( b >> 5 );                                               \
      a -= b; a -= c; a ^= ( c >> 3 );                                               \
      b -= c; b -= a; b ^= ( a << 10 );                                              \
      c -= a; c -= b; c ^= ( b >> 15 );                                              \
    } while (0)

    # define HASH_JEN(key,keylen,hashv)                                              \
    do {                                                                             \
      unsigned _hj_i,_hj_j,_hj_k;                                                    \
      unsigned const char *_hj_key=(unsigned const char*)(key);                      \
      hashv = 0xfeedbeefu;                                                           \
      _hj_i = _hj_j = 0x9e3779b9u;                                                   \
      _hj_k = (unsigned)(keylen);                                                    \
      while (_hj_k >= 12U) {                                                         \
        _hj_i +=    (_hj_key[0] + ( (unsigned)_hj_key[1] << 8 )                      \
            + ( (unsigned)_hj_key[2] << 16 )                                         \
            + ( (unsigned)_hj_key[3] << 24 ) );                                      \
        _hj_j +=    (_hj_key[4] + ( (unsigned)_hj_key[5] << 8 )                      \
            + ( (unsigned)_hj_key[6] << 16 )                                         \
            + ( (unsigned)_hj_key[7] << 24 ) );                                      \
        hashv += (_hj_key[8] + ( (unsigned)_hj_key[9] << 8 )                         \
            + ( (unsigned)_hj_key[10] << 16 )                                        \
            + ( (unsigned)_hj_key[11] << 24 ) );                                     \
                                                                                     \
         HASH_JEN_MIX(_hj_i, _hj_j, hashv);                                          \
                                                                                     \
         _hj_key += 12;                                                              \
         _hj_k -= 12U;                                                               \
      }                                                                              \
      hashv += (unsigned)(keylen);                                                   \
      switch ( _hj_k ) {                                                             \
        case 11: hashv += ( (unsigned)_hj_key[10] << 24 ); /* FALLTHROUGH */         \
        case 10: hashv += ( (unsigned)_hj_key[9] << 16 );  /* FALLTHROUGH */         \
        case 9:  hashv += ( (unsigned)_hj_key[8] << 8 );   /* FALLTHROUGH */         \
        case 8:  _hj_j += ( (unsigned)_hj_key[7] << 24 );  /* FALLTHROUGH */         \
        case 7:  _hj_j += ( (unsigned)_hj_key[6] << 16 );  /* FALLTHROUGH */         \
        case 6:  _hj_j += ( (unsigned)_hj_key[5] << 8 );   /* FALLTHROUGH */         \
        case 5:  _hj_j += _hj_key[4];                      /* FALLTHROUGH */         \
        case 4:  _hj_i += ( (unsigned)_hj_key[3] << 24 );  /* FALLTHROUGH */         \
        case 3:  _hj_i += ( (unsigned)_hj_key[2] << 16 );  /* FALLTHROUGH */         \
        case 2:  _hj_i += ( (unsigned)_hj_key[1] << 8 );   /* FALLTHROUGH */         \
        case 1:  _hj_i += _hj_key[0];                      /* FALLTHROUGH */         \
        default: ;                                                                   \
      }                                                                              \
      HASH_JEN_MIX(_hj_i, _hj_j, hashv);                                             \
    } while (0)

    static inline int probe_start(const void * p, int mask) {
        # if 0
        uintptr_t v = reinterpret_cast<uintptr_t>(p) >> 3;
        return static_cast<int>(static_cast<unsigned>(v * 2654435761u) & mask);
        # else
        // See fig 6.6 of "Efficient use of task-based parallelism in HPC applications"
        unsigned h;
        HASH_JEN(&p, sizeof(void *), h);
        return static_cast<int>(h & static_cast<unsigned>(mask));
        # endif
    }

    Slot * raw_probe(const void * key) const {
        const int mask = _cap - 1;
        int h = probe_start(key, mask);
        for (;;) {
            Slot * s = _slots + h;
            if (!s->occupied || s->key == key) return s;
            h = (h + 1) & mask;
        }
    }

    void rehash(int new_cap) {
        Slot * old     = _slots;
        int    old_cap = _cap;
        _slots = static_cast<Slot *>(calloc(new_cap, sizeof(Slot)));
        assert(_slots);
        _cap = new_cap;
        _sz  = 0;
        for (int i = 0; i < old_cap; ++i) {
            Slot * o = old + i;
            if (!o->occupied) continue;
            int h = probe_start(o->key, _cap - 1);
            while (_slots[h].occupied) h = (h + 1) & (_cap - 1);
            Slot * ns    = _slots + h;
            ns->key      = o->key;
            ns->occupied = true;
            V * ov = reinterpret_cast<V *>(o->val);
            new (ns->val) V(std::move(*ov));
            ov->~V();
            ++_sz;
        }
        free(old);
    }

public:
    explicit FlatPtrMap(int initial_cap) {
        int c = 1;
        while (c < initial_cap * 2) c <<= 1;  // power-of-two, load <= 0.5
        _slots = static_cast<Slot *>(calloc(c, sizeof(Slot)));
        assert(_slots);
        _cap = c;
    }

    ~FlatPtrMap() {
        if (_slots) {
            for (int i = 0; i < _cap; ++i)
                if (_slots[i].occupied)
                    reinterpret_cast<V *>(_slots[i].val)->~V();
            free(_slots);
        }
    }

    // Returns {ptr-to-value, inserted?}. Default-constructs V on first insert.
    // MAY REHASH — all previously obtained V* become invalid after this call.
    std::pair<V *, bool> try_emplace(const void * key) {
        if (__builtin_expect(_sz * 2 >= _cap, 0))
            rehash(_cap * 2);
        Slot * s = raw_probe(key);
        if (s->occupied)
            return {reinterpret_cast<V *>(s->val), false};
        s->key = key; s->occupied = true;
        new (s->val) V();
        ++_sz;
        return {reinterpret_cast<V *>(s->val), true};
    }

    // Returns nullptr if not found. Never rehashes — always safe to call.
    inline V * find(const void * key) const {
        Slot * s = raw_probe(key);
        return s->occupied ? reinterpret_cast<V *>(s->val) : nullptr;
    }
};

class DependencyMap : public DependencyDomain
{
    struct Node
    {
        small_vector_t<access_t *> last_conc_writes;
        small_vector_t<access_t *> last_seq_reads;
        access_t *  last_seq_write;

    }; /* Node */

    FlatPtrMap<Node> map;

    public:
        explicit DependencyMap() : map(4096) {}
        ~DependencyMap() = default;

    public:

        //  access type         depend on
        //  SEQ-R               SEQ-W, CNC-W,  COM-W
        //  CNC-W               SEQ-R, SEQ-W,  COM-W,
        //  COM-W               SEQ-R, SEQ-W, (COM-W), CNC-W
        //  SEQ-W               SEQ-R, SEQ-W,  COM-W,  CNC-W

        void link(runtime_t * runtime, access_t * access);
        void put(access_t * access);

};

XKRT_NAMESPACE_END

#endif /* __DEPENDENCY_MAP_HPP__ */
