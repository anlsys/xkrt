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

#ifndef __XKRT_MEMORY_POOL_H__
# define __XKRT_MEMORY_POOL_H__

# include <vector>
# include <memory>
# include <functional>

/**
 *  A memory pool of object.
 *  This can be seen as a vector, where reallocs increase by fix-size, and
 *  where addresses of previously pushed elements won't change.
 */

template<typename T, size_t CHUNK_SIZE = 256>
struct memory_pool_t
{
    private:
        struct chunk_t
        {
            T * data;
            size_t current;

            chunk_t(void) :
                data(static_cast<T*>(::operator new[](sizeof(T) * CHUNK_SIZE))),
                current(0)
            {}

            ~chunk_t(void)
            {
                ::operator delete[](data);
            }

            // non-copyable, movable
            chunk_t(const chunk_t &) = delete;
            chunk_t & operator=(const chunk_t &) = delete;
            chunk_t(chunk_t && o) : data(o.data) { o.data = nullptr; }
        };

        std::vector<chunk_t> chunks;

    public:

        T *
        put(void)
        {
            if (chunks.empty() || chunks.back().current == CHUNK_SIZE)
                chunks.emplace_back();
            chunk_t & chunk = chunks.back();
            return chunk.data + chunk.current++;
        }

        template<typename... Args>
        T *
        put(Args && ... args)
        {
            return std::construct_at(this->put(), std::forward<Args>(args)...);
        }

        T *
        get(size_t index)
        {
            return chunks[index / CHUNK_SIZE].data + (index % CHUNK_SIZE);
        }

        size_t
        size(void) const
        {
            if (chunks.empty()) return 0;
            return (chunks.size() - 1) * CHUNK_SIZE + chunks.back().current;
        }

        void
        foreach(std::function<void(T *)> fn)
        {
            for (size_t ci = 0; ci < chunks.size(); ++ci)
            {
                chunk_t & chunk = chunks[ci];
                for (size_t i = 0; i < chunk.current ; ++i)
                    fn(chunk.data + i);
            }
        }

        void
        release(void)
        {
            this->chunks.clear();
        }

        /* iterator */
        struct iterator
        {
            memory_pool_t * pool;
            size_t          index;

            iterator(memory_pool_t * pool, size_t index) : pool(pool), index(index) {}

            T & operator*()  { return *pool->get(index); }
            T * operator->() { return  pool->get(index); }

            iterator & operator++() { ++index; return *this; }

            bool operator!=(const iterator & o) const { return index != o.index; }
            bool operator==(const iterator & o) const { return index == o.index; }
        };

        iterator begin() { return { this, 0 }; }
        iterator end()   { return { this, size() }; }

};

#endif /* __XKRT_MEMORY_POOL_H__ */
