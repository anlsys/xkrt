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

#ifndef __MEMORY_TREE_HPP__
# define __MEMORY_TREE_HPP__

# include <xkrt/support.h>
# include <xkrt/internals.h>
# include <xkrt/memory/access/common/khp-tree.hpp>

//  TODO : the design of this is terrible with a cyclic ownership with
//  'runtime_t' Redesign me !! This should be fully independent with
//  callbacks that can be parametrized, raised with global device ids

# include <xkrt/logger/logger.h>
# include <xkrt/logger/todo.h>
# include <xkrt/memory/access/coherency-controller.hpp>
# include <xkrt/sync/bits.h>
# include <xkrt/sync/lockable.hpp>

# pragma message(TODO "Replace the lock with a rwlock + atomic operations on block states. The global lock on the entire structure may be limiting scalability on multiple gpus")

# include <xkrt/runtime.h>              // this should gtfo
# include <xkrt/memory/area.h>          // this should gtfo
# include <xkrt/task/task.hpp>          // this should gtfo

# include <xkrt/memory/access/mode.h>

# include <algorithm>  // std::sort
# include <cstdint>
# include <functional>
# include <numeric> // std::iota

XKRT_NAMESPACE_BEGIN

/*
 *  Set to '1' to enable the following heuristic :
 *      If some memory is not coherent on any devices, but a device A is already
 *      fetching it, then if a device B wants to fetch concurrently, it defer
 *      the fetch to a D2D A->B submitted when A fetched.
 *
 *  Set to '0' so multiple H2D are submitted concurrently
 *
 *  This should be reworked to have a smarter routing mechanism too
 *  For instance, on Aurora:
 *      - H2D = PCIe5 = 50GB/s on 1 tile = 320GB/s if every tile are concurrently fetching
 *      - D2D (tile interconnect) = ~200GB/s
 *      - D2D (gpu xelink) = ~20GB/s
 *  We need some way to decide which link to use.
 *  For instance, MPICH does a round-robin between all links - in our case, we probably want to sature the PCI, and then forward instead
 */
# define USE_D2D_FORWARDING 1

# pragma message(TODO "Memory allocation is currently performed within a critical section... If memory eviction must be performed, this creates double-locking + a lot of time spent in the critical section. Reason is : we need a partition (in the memory tree) of the access to write the allocation information on each block of the partition")

# pragma message(TODO "'fetch' implementation should be optimize by reducing critical sections to the minimum number of commands. We could also consider making the structure lock-free but im concerned of actual performances (will lead to a lot of false-sharing...)")

# define MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX   (8)
# define MEMORY_REPLICATE_ALLOCATION_VIEW_NONE   (MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX)

typedef uint8_t memory_allocation_view_id_t;
static_assert(MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX <= (1 << (sizeof(memory_allocation_view_id_t)*8)));

typedef uint8_t memory_allocation_view_id_bitfield_t;
static_assert(MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX <= sizeof(memory_allocation_view_id_bitfield_t) * 8);

/* a forward request */
template <int K>
class KMemoryForward {

    public:
        using Rect = KHyperrect<2>;

    public:

        /* the access that requested the forward */
        access_t * access;

        /* dst chunk */
        area_chunk_t * chunk;

        /* the dst rect */
        Rect dst_hyperrect;

        /* the dst device */
        device_global_id_t device_global_id;

        /* the dst view */
        memory_replica_view_t device_view;

    public:

        KMemoryForward(
            access_t * access,
            area_chunk_t * chunk,
            const Rect & dst_hyperrect,
            device_global_id_t device_global_id,
            memory_replica_view_t & device_view
        ) :
            access(access),
            chunk(chunk),
            dst_hyperrect(dst_hyperrect),
            device_global_id(device_global_id),
            device_view(device_view)
        {}

        ~KMemoryForward() {}

}; /* KMemoryForward */

/* a view of memory allocation */
template <int K>
class KMemoryReplicaAllocationView {

    using MemoryForward = KMemoryForward<K>;

    public:

        /* the device memory chunk */
        area_chunk_t * chunk;

        /* the address of that view in [allocation, allocation + allocation->size[ */
        memory_replica_view_t view;

        /* awaiting operations */
        struct {

            /* tasks awaiting on that view to be transfered */
            std::vector<access_t *> accesses;

            /* must forward this view to other views using D2D transfer */
            std::vector<MemoryForward> forwards;

        } awaiting;

    public:

        KMemoryReplicaAllocationView(
            area_chunk_t * chunk,
            const uintptr_t addr,
            const size_t ld
        ) :
            chunk(chunk),
            view(addr, ld),
            awaiting()
        {
            ++(chunk->use_counter);
        }

        virtual ~KMemoryReplicaAllocationView() {}

}; /* MemoryReplicaAllocationView */

// if this assertion fails, many bitwise operation in the runtime will be wrong as
// they are implicitly done on int32 : (1 << device_global_id) will be an int -
// should update the runtime with (1UL << device_global_id) - maybe use a macro
// for 'one' depending on that size
static_assert(sizeof(memory_allocation_view_id_bitfield_t) * 8 <= 32);

/* a host replica on a device */
template <int K>
class KMemoryReplica
{
    using MemoryReplicaAllocationView = KMemoryReplicaAllocationView<K>;

    public:

        /* List of allocations for this device replica.
         * A device may have several allocations for the same 'host memory'
         * For instance, in the following case scenario where blocks are read in order
         *  ._______________________.
         *  |           |           |
         *  |    (1)    |    (2)    |
         *  |___________|___________|
         *  |           |           |
         *  |    (3)    |    (4)    |
         *  .___________|___________.
         *
         *  - (1)           - read a tile               (allocation 1)
         *  - (2)           - read a tile               (allocation 2)
         *  - (3)           - read a tile               (allocation 3)
         *  - (4)           - read a tile               (allocation 4)
         *  - (1,2,3,4)     - read all tiles at once    (no continuous allocation...)
         *
         *  As BLAS requires a single continuous allocation per matrix, we are
         *  fucked and have to reallocate on the 5-th access
         *
         *  The 'MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX' controls how many
         *  allocations of the same data may exists at most
         */

        /* array of allocations */
        MemoryReplicaAllocationView * allocations[MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX];
        volatile memory_allocation_view_id_t nallocations;

        /* coherent allocations */
        volatile memory_allocation_view_id_bitfield_t coherency;

        /* fetching allocations */
        volatile memory_allocation_view_id_bitfield_t fetching;

        static_assert(sizeof(memory_allocation_view_id_bitfield_t) * 8 >= MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX);

    public:
        KMemoryReplica() : allocations(), nallocations(0), coherency(0), fetching(0) {}
        KMemoryReplica(const KMemoryReplica & r)
        {
            (void) r;
            LOGGER_FATAL("Implement copy constructor");
        }
        ~KMemoryReplica() {}

}; /* MemoryReplica */

/* a memory block, one per tree node */
template <int K>
class KMemoryBlock {

    using Rect = KHyperrect<K>;
    using MemoryReplica = KMemoryReplica<K>;
    using MemoryReplicaAllocationView = KMemoryReplicaAllocationView<K>;

    public:

        /* per device replica info */
        MemoryReplica replicas[XKRT_DEVICES_MAX];

        /* coherent devices (i.e. devices with at least one coherent allocation) */
        volatile device_global_id_bitfield_t coherency;

        /* fetching devices (i.e. devices with at least one fetching allocation) */
        volatile device_global_id_bitfield_t fetching;

        # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
        /* true/false whether the block is registered */
        bool registered;
        # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

    public:

        /* a new memory block, assume it is coherent on the host */
        KMemoryBlock() :
            replicas(),
            coherency(0),
            fetching(0)
            # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
            , registered(false)
            # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */
        {}

        inline void
        memory_block_init(
            const Rect & block_rect,
            const KMemoryBlock & inheriting_block,
            const Rect & inheriting_rect,
            const size_t sizeof_type
        ) {
            /////////////////////////////////
            //  HOST_VIEW HAS TO BE OFFSET //
            /////////////////////////////////
            INTERVAL_DIFF_TYPE_T d[K];
            Rect::distance_manhattan(inheriting_rect, block_rect, d);

            //////////////////////////////////
            //  DUPPLICATE REPLICATE INFOS  //
            //////////////////////////////////

            for (device_global_id_t device_global_id = 0 ; device_global_id < XKRT_DEVICES_MAX ; ++device_global_id)
            {
                // retrieve this device replica
                      MemoryReplica *            replica =            this->replicas + device_global_id;
                const MemoryReplica * inheriting_replica = inheriting_block.replicas + device_global_id;

                // dupplicate allocations
                replica->nallocations = inheriting_replica->nallocations;
                for (memory_allocation_view_id_t i = 0 ; i < inheriting_replica->nallocations ; ++i)
                {
                    const MemoryReplicaAllocationView * inheriting_allocation = inheriting_replica->allocations[i];

                    // warning: 'ld' here depends on the allocation itself
                    const INTERVAL_DIFF_TYPE_T offset   = d[ACCESS_BLAS_ROW_DIM] + d[ACCESS_BLAS_COL_DIM] * inheriting_allocation->view.ld * sizeof_type;
                    const uintptr_t begin_addr          = (uintptr_t) ((INTERVAL_DIFF_TYPE_T) inheriting_allocation->view.addr + offset);
                    assert(begin_addr >= inheriting_allocation->chunk->ptr);

                    # pragma message(TODO "This memory is currently leaked when 'invalidate' is called")
                    MemoryReplicaAllocationView * allocation = new MemoryReplicaAllocationView(inheriting_allocation->chunk, begin_addr, inheriting_allocation->view.ld);
                    replica->allocations[i] = allocation;
                    // allocation->awaiting must remain empty, tasks will be notified through the shrinked block
                }

                // dupplicate fetching / coherency infos
                replica->fetching  = inheriting_replica->fetching;
                replica->coherency = inheriting_replica->coherency;
            }

            //////////////////////////////
            //  VALID BITS ARE STILL OK //
            //////////////////////////////

            this->coherency = inheriting_block.coherency;
            this->fetching = inheriting_block.fetching;

            # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
            this->registered = inheriting_block.registered;
            # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */
        }

        /* a block from splitting an existing one */
        KMemoryBlock(
            const Rect & block_rect,
            const KMemoryBlock & inheriting_block,
            const Rect & inheriting_rect,
            const size_t sizeof_type
        ) {
            static_assert(K == 2);
            this->memory_block_init(block_rect, inheriting_block, inheriting_rect, sizeof_type);
        }
        ~KMemoryBlock() {}

}; /* KMemoryBlock */

/* storage passed when searchingi n the tree */
template <int K>
class KBLASMemoryTreeNodeSearch {

    using Rect = KHyperrect<K>;
    using MemoryBlock = KMemoryBlock<K>;
    using MemoryForward = KMemoryForward<K>;

    public:
        class Partite {

            public:

                /* memory block in the tree (WARNING : this is mutable outside a 'lock' section) */
                MemoryBlock * block;

                /* The hyperrect of this block (intersection of the access with the tree node) */
                const Rect hyperrect;

                /* dst device */
                device_global_id_t dst_device_global_id;

                /* replica allocation to use as dst (in MemoryReplica::allocations) */
                memory_allocation_view_id_t dst_allocation_view_id;

                /* dst view */
                memory_replica_view_t dst_view;

                /* source device */
                device_global_id_t src_device_global_id;

                /* replica allocation to use as src (in MemoryReplica::allocations) */
                memory_allocation_view_id_t src_allocation_view_id;

                /* src view */
                memory_replica_view_t src_view;

                /* the source chunk for that partite */
                area_chunk_t * src_chunk;

                /* true if this block is already being fetched by a concurrent read access */
                bool must_fetch;

            public:

                Partite(MemoryBlock * b, const Rect & h) :
                    block(b),
                    hyperrect(h),
                    dst_device_global_id(HOST_DEVICE_GLOBAL_ID),
                    dst_allocation_view_id(MEMORY_REPLICATE_ALLOCATION_VIEW_NONE),
                    dst_view(),
                    src_device_global_id(HOST_DEVICE_GLOBAL_ID),
                    src_allocation_view_id(MEMORY_REPLICATE_ALLOCATION_VIEW_NONE),
                    src_view(),
                    must_fetch(true)
                {}

                virtual ~Partite() {}

                bool
                operator<(const Partite & p) const
                {
                    const bool is_left  = this->hyperrect[ACCESS_BLAS_COL_DIM].a < p.hyperrect[ACCESS_BLAS_COL_DIM].a;
                    const bool is_right = this->hyperrect[ACCESS_BLAS_COL_DIM].a > p.hyperrect[ACCESS_BLAS_COL_DIM].a;

                    if (is_left)
                        return true;

                    if (is_right)
                        return false;

                    // vertically aligned
                    assert(this->hyperrect[ACCESS_BLAS_COL_DIM].a == p.hyperrect[ACCESS_BLAS_COL_DIM].a);

                    const bool is_up    = this->hyperrect[ACCESS_BLAS_ROW_DIM].a < p.hyperrect[ACCESS_BLAS_ROW_DIM].a;
                    const bool is_down  = this->hyperrect[ACCESS_BLAS_ROW_DIM].a > p.hyperrect[ACCESS_BLAS_ROW_DIM].a;

                    if (is_up)
                        return true;

                    if (is_down)
                        return false;

                    // horizontally aligned
                    assert(this == &p);
                    return false;
                }

        }; /* Partite */

        class Partition {

            public:

                /* the partite of that partition */
                std::vector<Partite> partites;

                /* the dst chunk onto the device, where all partites should write disjointly */
                area_chunk_t * chunk;

            public:
                Partition() : partites(), chunk(nullptr) {}
                ~Partition() {}

                /* return the left-most and upper-most block of the partition */
                inline Partite &
                get_leftmost_uppermost_block(void)
                {
                    const size_t nblocks = this->partites.size();
                    size_t j = 0;

                    for (size_t i = 1 ; i < nblocks ; ++i)
                    {
                        const Partite & bi = this->partites[i];
                        const Partite & bj = this->partites[j];
                        if (bi < bj)
                            j = i;
                    }

                    return this->partites[j];
                }

        }; /* Partition */

   public:

       /* different search type */
        enum Type : uint8_t {
            INSERTING_BLOCKS     = 0,    // insert new blocks
            SEARCH_FOR_PARTITION = 1,    // search for a partition
            SEARCH_FETCHED       = 2,    // search tasks awaiting on blocks (to be transfered onto a gpu, typically)
            SEARCH_OWNERS        = 3,    // search how many bytes owns each device
            # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
            REGISTER             = 4,    // mark memory block as registered
            UNREGISTER           = 5,    // mark memory block as unregistered
            # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */
       };

   public:

       /////////////////////////////////
       // used in all types of search //
       /////////////////////////////////

       /* type of search performing */
       Type type;

        /* device global id, on which we are looking for coherent blocks or making coherent blocks */
       const device_global_id_t device_global_id;

       //////////////////////////////////////////////////////
       // used if type == INSERTING_BLOCKS //
       //////////////////////////////////////////////////////

       /* the access being inserted / intersected */
       access_t * access;

       ///////////////////////////////////////
       // used if type == SEARCH_FOR_PARTITION //
       ///////////////////////////////////////

       /*
        * list of blocks for the current access.
        * The set { b.rect / b in partition } is a partition of the space represented by access->hyperrect
        */
        Partition partition;

        ///////////////////////////////////////////
        // used if type == SEARCH_FETCHED //
        ///////////////////////////////////////////
        area_chunk_t * chunk;
        struct {
            std::vector<access_t *> accesses;
            std::vector<MemoryForward> forwards;
        } awaiting;

        ///////////////////////////////////
        // used if type == SEARCH_OWNERS //
        ///////////////////////////////////
        size_t bytes_owned[XKRT_DEVICES_MAX];

   public:
       KBLASMemoryTreeNodeSearch() : KBLASMemoryTreeNodeSearch(HOST_DEVICE_GLOBAL_ID) {}

       KBLASMemoryTreeNodeSearch(
           device_global_id_t devid
       ) :
           type(INSERTING_BLOCKS),
           device_global_id(devid),
           access(nullptr),
           partition(),
           chunk(nullptr),
           awaiting()
       {}

       virtual ~KBLASMemoryTreeNodeSearch() {}

       void
       prepare_insert(access_t * a)
       {
           this->access = a;
           this->type = INSERTING_BLOCKS;
       }

       void
       prepare_search_partition(void)
       {
           assert(this->partition.partites.size() == 0);
           this->partition.partites.clear();
           this->type = SEARCH_FOR_PARTITION;
       }

       void
       prepare_search_fetched(area_chunk_t * chunk)
       {
           this->chunk = chunk;
           this->type = SEARCH_FETCHED;
       }

       void
       prepare_search_owners(void)
       {
           memset(this->bytes_owned, 0, sizeof(this->bytes_owned));
           this->type = SEARCH_OWNERS;
       }

       void
       prepare(Type t)
       {
           this->type = t;
       }

}; /* KBLASMemoryTreeNodeSearch */

template <int K>
class KBLASMemoryTreeNode : public KHPTree<K, KBLASMemoryTreeNodeSearch<K>>::Node {

    using Base = typename KHPTree<K, KBLASMemoryTreeNodeSearch<K>>::Node;
    using Rect = KHyperrect<K>;
    using MemoryBlock = KMemoryBlock<K>;
    using MemoryReplica = KMemoryReplica<K>;
    using MemoryReplicaAllocationView = KMemoryReplicaAllocationView<K>;
    using Node = KBLASMemoryTreeNode<K>;
    using Partite = typename KBLASMemoryTreeNodeSearch<K>::Partite;
    using Search = KBLASMemoryTreeNodeSearch<K>;

    public:

        /* the memory block represented by this node */
        MemoryBlock block;

    public:

        /* the rect was never accessed before, create a new node */
        KBLASMemoryTreeNode<K>(
            const access_t * access,
            const Rect & r,
            const int k,
            const Color color
        ) :
            Base(r, k, color),
            block()
        {
            (void) access;
        }

        /**
         * A new node is being created from a split, make it inherit its original node 'src'
         *  - access - the access
         *  - r - the shrinked rect that this is inheriting from
         *  - k - the dimension that got splitted
         *  - color - the node color
         *  - src - the node that got split
         *
         * We have:
         *  U (src->hyperrect, r) == the node rect before being shrinked
         *  n (src->hyperrect, r) = {} - empty intersection
         */
        KBLASMemoryTreeNode<K>(
            const Rect & r,
            const int k,
            const Color color,
            const Node * src,
            const size_t sizeof_type
        ) :
            Base(r, k, color),
            block(r, src->block, src->hyperrect, sizeof_type)
        {}

    public:

        void
        dump_str(FILE * f) const
        {
            Base::dump_str(f);
        }

        void
        dump_hyperrect_str(FILE * f) const
        {
            // Base::dump_hyperrect_str(f);

            for (device_global_id_t device_global_id = 0 ; device_global_id < XKRT_DEVICES_MAX ; ++device_global_id)
            {
                const int devbit = (1 << device_global_id);
                fprintf(f, "\\\\ dev %d - coherent=%d",
                    device_global_id,
                    this->block.coherency & devbit ? 1 : 0
                );
            }
        }

}; /* KBLASMemoryTreeNode */

template <int K>
class KBLASMemoryTree : public KHPTree<K, KBLASMemoryTreeNodeSearch<K>>, public Lockable, public MemoryCoherencyController {

    public:
        using Base = KHPTree<K, KBLASMemoryTreeNodeSearch<K>>;
        using Rect = KHyperrect<K>;
        using MemoryBlock = KMemoryBlock<K>;
        using MemoryForward = KMemoryForward<K>;
        using MemoryReplica = KMemoryReplica<K>;
        using MemoryReplicaAllocationView = KMemoryReplicaAllocationView<K>;
        using Node = KBLASMemoryTreeNode<K>;
        using NodeBase = typename KHPTree<K, KBLASMemoryTreeNodeSearch<K>>::Node;
        using Partite = typename KBLASMemoryTreeNodeSearch<K>::Partite;
        using Partition = typename KBLASMemoryTreeNodeSearch<K>::Partition;
        using Search = KBLASMemoryTreeNodeSearch<K>;

    public:

        KBLASMemoryTree(
            runtime_t * runtime,
            const size_t ld,
            const size_t sizeof_type,
            const bool merge_transfers
        ) :
            Base(),
            runtime(runtime),
            ld(ld),
            sizeof_type(sizeof_type),
            merge_transfers(merge_transfers),
            pagesize(getpagesize())
        {}

        ~KBLASMemoryTree() {}

        /* the runtime so the tree can launch data movements */
        runtime_t * runtime;

        /* the router */
        Router * router;

        /* the ld used in that memory tree */
        const size_t ld;

        /* the size of type */
        const size_t sizeof_type;

        /* whether transfers in continuous virtual memory should be merged */
        const bool merge_transfers;

        /* pagesize, to avoid repetitively calling `getpagesize()` */
        const size_t pagesize;

    public:

        typedef struct  fetch_t
        {
            /* the rectangle of that fetch */
            Rect rect;

            /* the host memory view */
            memory_view_t host_view;

            /* src device id */
            device_global_id_t src_device_global_id;

            /* src view */
            memory_replica_view_t src_view;

            /* mark 'fetched' all the tasks awaiting on that allocation */
            area_chunk_t * dst_chunk;

            /* dst device id */
            device_global_id_t dst_device_global_id;

            /* dst view */
            memory_replica_view_t dst_view;

            /* whether this fetch had been merged, and can therefore be skipped */
            bool merged;

        }               fetch_t;

        typedef struct  fetch_list_t
        {
            /* the memory tree */
            KBLASMemoryTree * tree;

            /* list of fetches to submit */
            fetch_t * fetches;

            /* 'fetches' capacity */
            task_wait_counter_type_t capacity;

            /* number of 'fetches' set */
            task_wait_counter_type_t n;

            /* number of pending fetches */
            volatile task_wait_counter_t pending;

            fetch_list_t(KBLASMemoryTree * tree, fetch_t * fetches, task_wait_counter_type_t capacity) : tree(tree), fetches(fetches), capacity(capacity), n(0), pending(0) {}
            ~fetch_list_t() {}

            fetch_t *
            prepare_next_fetch(void)
            {
                fetch_t * fetch = this->fetches + this->n;
                ++this->n;
                assert(this->n <= this->capacity);
                return fetch;
            }

            /* the list can be deleted if this returns '0' */
            task_wait_counter_type_t
            fetched(task_wait_counter_type_t decr = 1)
            {
                const task_wait_counter_type_t p = this->pending.fetch_sub(decr, std::memory_order_relaxed);
                assert(p >= 0);
                return p;
            }

            void
            fetching(task_wait_counter_type_t inc = 1)
            {
                this->pending.fetch_add(inc, std::memory_order_relaxed);
            }

        }               fetch_list_t;

        # if 0
        /* if merging is enabled, merge consecutive transfers to a single transfer */
        static inline void
        fetch_list_reduce(fetch_list_t * list)
        {
            /* fast way out */
            const size_t n = list->n;
            if (n <= 1)
                return ;

           /**
            *  Given two regions 'a' and 'b' we note 'a -> b' if 'a' and 'b'
            *  are consecutive in memory, with 'b' being right-after 'a'.
            *  We note a+b the merged (continuous) memory region.
            *
            *  Given a passed list of region
            *      L = [a, b, c, d, e]
            *  with a->c and e->d, this routine updates the list so that it becomes
            *      L = [a+c, b, e+d]
            *
            *  A fetch list element is not only a region, but also has a:
            *      - a source device
            *      - a destinatary device
            *  which needs to match between elements so they are merged.
            *
            *   Note at this point, the list is a partition of an access,
            *   so there cannot be overlap between represented regions
            *
            */

            /* array of 'n' fetches */
            fetch_t * fetches = (fetch_t *) (list + 1);

            /* create vector [0, 1, ..., n-1] */
            std::vector<int> indices(n);
            std::iota(indices.begin(), indices.end(), 0);

            // Sadly, we need to sort here.
            // Even though data are partially ordered in the KHP tree, the search does not necessarily creates a sorted list.
            // To remove the 'sort' here, the best way might be to implement a
            // 'sorted-search' in the KHP-Tree, that search intersecting nodes in-order (left-to-right, bottom-to-top)

            /* sort the vector so fetches[indices[i]] < fetches[indices[i+1]] in memory */
            std::sort(
                std::begin(indices),
                std::end(indices),
                [fetches](size_t i, size_t j) {
                    assert(i != j);
                    const fetch_t * fi = fetches + i;
                    const fetch_t * fj = fetches + j;
                    return fi->host_view.begin_addr() < fj->host_view.begin_addr();
                }
            );

            /* find continuous fetches and merge them */
            fetch_t * fi = fetches + indices[0];
            assert(fi->merged == false);
            for (size_t j = 1 ; j < n ; ++j)
            {
                fetch_t * fj = fetches + indices[j];
                assert(fi->host_view.sizeof_type == fj->host_view.sizeof_type);

                /* already merged */
                if (fj->merged)
                    continue ;

                /* fetches must occur between the same devices */
                if (    fi->src_device_global_id == fj->src_device_global_id    &&
                        fi->dst_device_global_id == fj->dst_device_global_id    &&
                        fi->dst_chunk            == fj->dst_chunk)
                {
                    /**
                     *  case (1)
                     *              n
                     *  ----------------->
                     *  |   |       |
                     *  |   |_ _ _ _|
                     *  |
                     *  |  _ _ _ _
                     *  | |_ _ _ _|
                     *  v
                     *
                     *  or
                     *              n
                     *  ----------------->
                     *  |  _ _ _ _
                     *  | |       |
                     *  | |_ _ _ _|
                     *  | |_______|
                     *  |
                     *  v
                     *
                     *  or
                     *              n
                     *  ----------------->
                     *  |  _ _ _ _ _ _
                     *  | |       |   |
                     *  | |_ _ _ _|_ _|
                     *  |
                     *  v
                     */
                    const size_t dm = fi->host_view.m;
                    const size_t dn = fj->host_view.n;
                    if (fi->host_view.ld == fj->host_view.ld && (fi->host_view.offset_addr(dm,  0) == fj->host_view.begin_addr() || fi->host_view.offset_addr( 0, dn) == fj->host_view.begin_addr()))
                    {
                        assert(!fi->rects[0].is_empty());
                        assert(!fj->rects[0].is_empty());
                        assert( fi->rects[1].is_empty());
                        assert( fj->rects[1].is_empty());

                        matrix_from_rects(
                            fi->host_view,
                            fi->rects[0], fj->rects[0],
                            fi->host_view.ld,
                            fi->host_view.sizeof_type
                        );

                        fi->rects[1] = fj->rects[0];

                        fj->merged = true;
                        list->fetched();
                    }
                    /* else, try to merge next fetches */
                    else
                    {
                        assert(fj->merged == false);
                        fi = fj;
                    }
                }
            }
        }
        # endif

        static inline void
        fetch_callback_access(
            runtime_t * runtime,
            fetch_t * fetch,
            access_t * access
        ) {
            assert(access->task);
            LOGGER_DEBUG("task `%s` fetched `%p` on device `%u`",
                    access->task->label, (void *) (fetch->dst_chunk ? fetch->dst_chunk->ptr : NULL), fetch->dst_device_global_id);
            access->state = ACCESS_STATE_FETCHED;

            device_t * device = runtime->device_get(fetch->dst_device_global_id);
            assert(device);
            __task_fetched(1, access->task, task_execute, runtime, device);
        }

        static inline fetch_list_t *
        fetch_list_new(KBLASMemoryTree * tree, task_wait_counter_type_t capacity)
        {
            fetch_list_t * list = (fetch_list_t *) calloc(1, sizeof(fetch_list_t) + capacity * sizeof(fetch_t));
            assert(list);
            new (list) fetch_list_t(tree, (fetch_t *) (list + 1), capacity);
            return list;
        }

        static void
        fetch_callback(void * args[XKRT_CALLBACK_ARGS_MAX])
        {
            assert(XKRT_CALLBACK_ARGS_MAX >= 4);

            runtime_t * runtime = (runtime_t *) args[0];
            assert(runtime);

            access_t * access = (access_t *) args[1];
            assert(access);

            fetch_list_t * list = (fetch_list_t *) args[2];
            assert(list);

            KBLASMemoryTree * tree = list->tree;
            assert(tree);

            task_wait_counter_type_t fetch_idx = static_cast<task_wait_counter_type_t>(reinterpret_cast<uintptr_t>(args[3]));
            assert(fetch_idx >= 0 && fetch_idx < list->n);

            fetch_t * fetch = list->fetches + fetch_idx;

            if (fetch->dst_chunk)
                LOGGER_DEBUG("Fetch completed for allocation `%p`", (void *) fetch->dst_chunk->ptr);

            // avoid early deletion if a 'reset' is called before returning from this
            tree->ref();
            {
                fetch_callback_access(runtime, fetch, access);

                /* `fetch->dst_chunk` is the allocated memory chunk on which the data had been fetched. */
                assert(fetch->dst_chunk || fetch->dst_device_global_id == HOST_DEVICE_GLOBAL_ID);

                /* Search in the tree to unmark the block 'fetching' bit, and forward data to awaiting tasks using D2D */
                Search search(fetch->dst_device_global_id);
                search.prepare_search_fetched(fetch->dst_chunk);
                tree->lock();
                {
                    tree->intersect(search, fetch->rect);
                }
                tree->unlock();

                if (fetch->dst_device_global_id != HOST_DEVICE_GLOBAL_ID)
                {
                    /* callback to release awaiting tasks */
                    for (access_t * & access_awaiting : search.awaiting.accesses)
                        fetch_callback_access(runtime, fetch, access_awaiting);

                    /* callback to forward the data to other devices */
                    task_wait_counter_type_t nforwards = (task_wait_counter_type_t) search.awaiting.forwards.size();
                    if (nforwards)
                    {
                        fetch_list_t * forward_list = fetch_list_new(tree, nforwards);

                        # pragma message(TODO "Forwards consecutive in memory should be merged")
                        for (task_wait_counter_type_t i = 0 ; i < nforwards ; ++i)
                        {
                            MemoryForward & forward = search.awaiting.forwards[i];

                            assert(forward.access);
                            assert(forward.chunk);

                            fetch_t * forward_fetch = forward_list->prepare_next_fetch();
                            assert(fetch);
                            assert(i == forward_list->n - 1);
                            forward_list->fetching();

                            // upcoming copy only needs m, n, sizeof_type
                            matrix_from_rect(forward_fetch->host_view, forward.dst_hyperrect, tree->ld, tree->sizeof_type);

                            // the chunk to use
                            forward_fetch->dst_chunk = forward.chunk;

                            // the dst device to forward to
                            forward_fetch->dst_device_global_id = forward.device_global_id;

                            // the forward dst view - memory region where to forward the data
                            forward_fetch->dst_view = forward.device_view;

                            // only 1 rect representing the forward view
                            forward_fetch->rect = forward.dst_hyperrect;

                            // the just-fetched 'dst' is the new 'src'
                            forward_fetch->src_device_global_id = fetch->dst_device_global_id;

                            // this fetch maybe got a larger region than the one to forward, for instance
                            //      fetch->dst = [                                                      ]
                            //  while maybe
                            //    forward->dst =                    [                   ]
                            // in such case, we only need to forward the sub-region.
                            //
                            // the just-fetched 'dst' is the new 'src' - offset it to the begining of the forward view
                            assert(fetch->host_view.begin_addr() <= forward_fetch->host_view.begin_addr());
                            const size_t offset = forward_fetch->host_view.begin_addr() - fetch->host_view.begin_addr();
                            new (&forward_fetch->src_view) memory_replica_view_t(fetch->dst_view.addr + offset, fetch->dst_view.ld);

                            // can launch the forward already
                            tree->fetch_list_launch_ith(forward.access, forward_list, i);

                            // no need to reduce the list, we already have only 1 copy per dst device
                        }
                    }
                } /* if != HOST_DEVICE_GLOBAL_ID */
                list->fetched();
            } /* tree->ref() */
            tree->unref();
        }

        ////////////////////////////////////////////////////////////
        // Create a list of fetch requests for the given accesses //
        ////////////////////////////////////////////////////////////

        /* convert a partition to a minimal fetch list, merging consecutive partite to a single transfer */
        fetch_list_t *
        fetch_list_from_partition(
            Partition & partition
        ) {
            task_wait_counter_type_t capacity = (task_wait_counter_type_t) partition.partites.size();
            if (capacity == 0)
                return NULL;

            fetch_list_t * list = fetch_list_new(this, capacity);
            assert(list);

            for (Partite & partite : partition.partites)
            {
                if (!partite.must_fetch)
                    continue ;

                /* one replica must be non-null (a null replica means to use the host view) */
                assert(partite.dst_allocation_view_id != MEMORY_REPLICATE_ALLOCATION_VIEW_NONE ||
                        partite.src_allocation_view_id != MEMORY_REPLICATE_ALLOCATION_VIEW_NONE);

                /* set the views */
                memory_view_t host_view;
                matrix_from_rect(host_view, partite.hyperrect, this->ld, this->sizeof_type);
                const memory_replica_view_t host_replica_view(host_view.begin_addr(), this->ld);
                const memory_replica_view_t dst_view = (partite.dst_allocation_view_id == MEMORY_REPLICATE_ALLOCATION_VIEW_NONE) ? host_replica_view : partite.dst_view;
                const memory_replica_view_t src_view = (partite.src_allocation_view_id == MEMORY_REPLICATE_ALLOCATION_VIEW_NONE) ? host_replica_view : partite.src_view;

                /* allocate fetch info for the callback argument */
                fetch_t * fetch = list->prepare_next_fetch();
                fetch->rect                 = partite.hyperrect;
                fetch->host_view            = host_view;
                fetch->src_device_global_id = partite.src_device_global_id;
                fetch->src_view             = src_view;
                fetch->dst_chunk            = partition.chunk;
                fetch->dst_device_global_id = partite.dst_device_global_id;
                fetch->dst_view             = dst_view;
            }
            list->fetching(list->n);

            return list;
        }

        void
        fetch_list_to_host_setup_partition(Partition & partition)
        {
            assert(this->is_locked());
            const device_global_id_bitfield_t devbit = (device_global_id_bitfield_t) (1 << HOST_DEVICE_GLOBAL_ID);

            /* launch fetch on each device */
            for (Partite & partite : partition.partites)
            {
                MemoryBlock * block = partite.block;

                /* we can skip the transfer if whether:
                 *  - the block is not coherent on any devices, then assume it is coherent on the host
                 *  - the host already has a coherent replica
                 *  - the host is already fetching
                 */
                if (block->coherency == 0 || (block->coherency & devbit) || (block->fetching & devbit))
                {
                    partite.must_fetch = false;
                    continue ;
                }
                block->fetching |= devbit;

                /////////////////////////
                // SRC - FIND BEST SRC //
                /////////////////////////

                // take first device with a coherent replica, and the first coherent allocation for all partites,
                // so continuous partites are on the same device and allocation
                // for merging them later

                // device_global_id_t src = __random_set_bit(partite.block->coherency) - 1;
                device_global_id_t src = (device_global_id_t) (__builtin_ffs(partite.block->coherency) - 1);
                assert(src >= 0);

                // Get a coherent allocation on that device
                MemoryReplica & src_replica = partite.block->replicas[src];
                assert(src_replica.nallocations > 0);
                assert(src_replica.coherency);

                # if 0
                // Heuristic : get the first coherent
                const memory_allocation_view_id_t src_allocation_view_id = (memory_allocation_view_id_t) (__builtin_ffs(src_replica.coherency) - 1);
                assert(0 <= src_allocation_view_id && src_allocation_view_id < src_replica.nallocations);
                # else
                // Heuristic: use the largest coherent allocation on that device to reduce D2D copies when trying to merge later
                memory_allocation_view_id_t src_allocation_view_id = 0;
                for (memory_allocation_view_id_t allocation_view_id = 1 ; allocation_view_id < src_replica.nallocations ; ++allocation_view_id)
                {
                    const MemoryReplicaAllocationView * src_allocation_view = src_replica.allocations[src_allocation_view_id];
                    const MemoryReplicaAllocationView *     allocation_view = src_replica.allocations[allocation_view_id];
                    if (src_allocation_view->chunk->size < allocation_view->chunk->size)
                        src_allocation_view_id = allocation_view_id;
                }
                # endif

                // retrieve and set src view infos
                assert(0 <= src_allocation_view_id && src_allocation_view_id < src_replica.nallocations);
                MemoryReplicaAllocationView * src_allocation_view = src_replica.allocations[src_allocation_view_id];
                assert(src_allocation_view);

                // set partite transfer infos
                partite.src_allocation_view_id  = src_allocation_view_id;
                partite.src_device_global_id    = src;
                partite.src_view                = src_allocation_view->view;
                partite.src_chunk               = src_allocation_view->chunk;

                partite.dst_allocation_view_id  = MEMORY_REPLICATE_ALLOCATION_VIEW_NONE;
                partite.dst_device_global_id    = HOST_DEVICE_GLOBAL_ID;
                // no need to set partite->dst_view - host view will be used
            }
        }

        /* create a list of fetch request to perform for the given rects */
        fetch_list_t *
        fetch_list_to_host(
            access_t * access
        ) {
            assert(access->type == ACCESS_TYPE_SEGMENT || access->type == ACCESS_TYPE_BLAS_MATRIX);

            /* optim: if there had been no devices accesses previously, nothing to do */
            if (this->root == NULL)
                return NULL;

            Search search(HOST_DEVICE_GLOBAL_ID);
            this->lock();
            {
                /* step (1) ensure the access is represented in the tree as blocks */
                search.prepare_insert(access);
                for (Rect & rect : access->rects())
                    this->insert(search, rect);

                /* step (2) find all blocks representing the access */
                search.prepare_search_partition();
                for (const Rect & rect : access->rects())
                    this->intersect(search, rect);
                assert(search.partition.partites.size() >= 1);

                /* step (5) if read access, find src/dst, and setup views to transfer on step (7) */
                this->fetch_list_to_host_setup_partition(search.partition);
            }
            this->unlock();

            /* generate the fetch list */
            return this->fetch_list_from_partition(search.partition);
        }

        ////////////////////////
        //  FETCH ON A DEVICE //
        ////////////////////////

        inline void
        fetch_access_allocate_eviction(
            device_global_id_t device_global_id,
            size_t size
        ) {

            /* adapted from 'memory_cache_evict_fromlist' */
            LOGGER_DEBUG("Evicting memory...");

            // TODO : currently deallocating as much as possible, maybe stop when there is a chunk big-enough of 'size'

            size_t freed = 0;
            auto f = [this, size, device_global_id, &freed](NodeBase * nodebase, void * args, bool & stop) {
                (void) args;

                Node * node = reinterpret_cast<Node *>(nodebase);
                assert(node);

                MemoryBlock & block = node->block;

                const device_global_id_bitfield_t devbit = (device_global_id_bitfield_t) (1 << device_global_id);

                const bool coherent_on_any_device        = block.coherency != 0;
                const bool coherent_on_device            = block.coherency &  devbit;
                const bool coherent_on_any_other_devices = block.coherency & ~devbit;

                MemoryReplica & replica = block.replicas[device_global_id];
                if (replica.fetching)
                    return ;

                if (!coherent_on_any_device || coherent_on_any_other_devices)
                {
                    /* evict all allocations */
                    for (int i = 0 ; i < replica.nallocations ; ++i)
                    {
                        MemoryReplicaAllocationView * allocation = replica.allocations[i];
                        assert(allocation);

                        /* if only this block uses the allocation */
                        if (allocation->chunk->use_counter == 1)
                        {
                            LOGGER_DEBUG("Evicted a block of size %zu MB", allocation->chunk->size/1024/1024);
                            this->runtime->memory_device_deallocate(device_global_id, allocation->chunk);
                            freed += allocation->chunk->size;
                        }
                        /* else: what to do ? */
                        else
                        {
                            // LOGGER_FATAL("Couldn't evict a chunk that is used in several allocation view - coherent_on_any_device=%d, coherent_on_any_other_devices=%d", coherent_on_any_device, coherent_on_any_other_devices);
                        }

                        // delete allocation;
                    }
                    replica.nallocations  = 0;
                    replica.coherency     = 0;
                    assert(replica.fetching == 0);

                    block.coherency &= (device_global_id_bitfield_t) ~devbit;

                    stop = freed >= 16*size;
                    // stop = false;
                }
                else if (coherent_on_device && replica.nallocations > 1)
                {
                    // TODO : only keep 1 coherent allocation
                    LOGGER_FATAL("coherent_on_device=%d, nallocations=%d", coherent_on_device, replica.nallocations);
                }
            };

            this->foreach_node_until(f, NULL);
        }

        inline area_chunk_t *
        fetch_access_allocate(
            access_t * access,
            device_global_id_t device_global_id
        ) {
            //////////////////////////
            // Allocate a new chunk //
            //////////////////////////

            const size_t size = access->host_view.m * access->host_view.n * access->host_view.sizeof_type;
            area_chunk_t * chunk = nullptr;
            int retry_cnt = 0;

            do {

                chunk = this->runtime->memory_device_allocate(device_global_id, size);
                if (chunk)
                    return chunk;

                // TODO : polling is risky here, because it may take a lock on the
                // memory tree, and 'memory_allocate' is called within a
                // memory-tree lock => double-lock deadlock

                // device_poll(device);
                fetch_access_allocate_eviction(device_global_id, size);

            } while (++retry_cnt < 32);

            LOGGER_FATAL("!! GPU IS OUT OF MEMORY !!");

            return nullptr;
        }

        /* Create a view for each partite of the partition, for the newly allocated chunk */
        inline void
        fetch_access_create_allocation_views(
            access_t * access,
            device_global_id_t device_global_id,
            Partition & partition,
            area_chunk_t * chunk
        ) {
            assert(chunk);

            /* allocate continuous memory for that access */
            # pragma message(TODO "Can we manage row/col major in a better way ? hardcoded col major here for cuda")
            const size_t          ld = access->host_view.m;            // cuda is col major
            const size_t sizeof_type = access->host_view.sizeof_type;

            /* retrieve upper left corner */
            const Partite & corner = partition.get_leftmost_uppermost_block();

            /* add a view */
            for (Partite & partite : partition.partites)
            {
                /* compute distance from corner */
                INTERVAL_DIFF_TYPE_T d[K];
                Rect::distance_manhattan(corner.hyperrect, partite.hyperrect, d);
                if (d[ACCESS_BLAS_ROW_DIM] < 0)
                {
                    d[ACCESS_BLAS_ROW_DIM] += this->ld * this->sizeof_type;
                    d[ACCESS_BLAS_COL_DIM] -= 1;
                }
                assert(d[ACCESS_BLAS_ROW_DIM] >= 0);
                assert(d[ACCESS_BLAS_COL_DIM] >= 0);

                const uintptr_t offset = d[ACCESS_BLAS_ROW_DIM] + d[ACCESS_BLAS_COL_DIM]*ld*sizeof_type;
                const uintptr_t begin_addr = chunk->ptr + offset;

                MemoryReplica & replica = partite.block->replicas[device_global_id];
                const memory_allocation_view_id_t allocation_view_id = replica.nallocations;
                replica.nallocations += 1;
                if (allocation_view_id >= MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX)
                    LOGGER_FATAL("Too many allocations of the same data on the same device... Increase `MEMORY_REPLICATE_ALLOCATION_VIEWS_MAX` and recompile");

                /* allocate the view of that block in the allocation */
                MemoryReplicaAllocationView * r = new MemoryReplicaAllocationView(chunk, begin_addr, ld);
                replica.allocations[allocation_view_id] = r;
                partite.dst_allocation_view_id = allocation_view_id;
            }
        }

        /* look for a continuous allocation that can store 'access' for the given partition */
        inline area_chunk_t *
        fetch_access_find_allocation_continuous(
            device_global_id_t device_global_id,
            Partition & partition
        ) {
            assert(this->is_locked());

            memory_allocation_view_id_t j = 0;
            int nallocations = partition.partites[0].block->replicas[device_global_id].nallocations;
            size_t nblocks = partition.partites.size();

            /* for each allocation of the block 0 */
            while (j < nallocations)
            {
                MemoryReplicaAllocationView * rj = partition.partites[0].block->replicas[device_global_id].allocations[j];

                /* for each other blocks */
                size_t i = 1;
                while (i < nblocks)
                {
                    /* for each allocation of other blocks */
                    int nallocations = partition.partites[i].block->replicas[device_global_id].nallocations;
                    for (memory_allocation_view_id_t k = 0 ; k < nallocations ; ++k)
                    {
                        /* this block has a view with the same allocation, check next block */
                        MemoryReplicaAllocationView * rk = partition.partites[i].block->replicas[device_global_id].allocations[k];
                        if (rj->chunk == rk->chunk)
                        {
                            partition.partites[i].dst_allocation_view_id = k;
                            goto next_block;
                        }
                    }

                    /* this block has no view view the same allocation, restart from the next view of block 0 */
                    goto next_view;

next_block:
                    ++i;
                    continue ;
                }

                /* every blocks have a view with the allocation 'allocation' */
                partition.partites[0].dst_allocation_view_id = j;
                return rj->chunk;

next_view:
                ++j;
                continue ;
            }
            return nullptr;
        }

        /* set 'partition.chunk' to a memory chunk and all 'partites.dst_allocation_view_id' to a view on that chunk */
        inline void
        fetch_access_find_allocation(
            access_t * access,
            device_global_id_t device_global_id,
            Partition & partition
        ) {
            assert(this->is_locked());

            /* lookfor a continuous allocation already existing for that access block partitioning */
            area_chunk_t * chunk = this->fetch_access_find_allocation_continuous(device_global_id, partition);
            if (chunk == nullptr)
            {
                /* no continuous allocation found, make a new one */
                LOGGER_DEBUG("No continuous allocation found, reallocating and creating a new view");
                chunk = this->fetch_access_allocate(access, device_global_id);
                assert(chunk);

                /* create new views */
                this->fetch_access_create_allocation_views(access, device_global_id, partition, chunk);
            }

            partition.chunk = chunk;
        }

        inline void
        fetch_access_setup_replicas(
            access_t * access,
            device_global_id_t device_global_id,
            Search & search
        ) {
            assert(this->is_locked());

            // we currently set the access view as the 'left-most' and 'upper-most' tile
            // (i.e with the smallest address - corresponding to the begining of this allocation)
            const Partite & partite = search.partition.get_leftmost_uppermost_block();
            assert(partite.dst_allocation_view_id != MEMORY_REPLICATE_ALLOCATION_VIEW_NONE);

            const MemoryReplicaAllocationView * r = partite.block->replicas[device_global_id].allocations[partite.dst_allocation_view_id];
            assert(r);

            access->device_view = r->view;
        }

        inline void
        fetch_access_setup_copies(
            access_t * access,
            device_global_id_t device_global_id,
            Partition & partition
        ) {
            assert(this->is_locked());
            assert(access->task);

            // if read mode is set
            if (access->mode & ACCESS_MODE_R)
            {
                const device_global_id_bitfield_t dst_devbit = (device_global_id_bitfield_t) (1 << device_global_id);

                // for each block of that access
                for (Partite & partite : partition.partites)
                {
                    ///////////////
                    // SETUP DST //
                    ///////////////

                    /* destinary allocation view id (cannot be host in this routine) */
                    const int dst_allocation_view_id = partite.dst_allocation_view_id;
                    assert(dst_allocation_view_id != MEMORY_REPLICATE_ALLOCATION_VIEW_NONE);
                    const memory_allocation_view_id_bitfield_t dst_allocbit = (memory_allocation_view_id_bitfield_t) (1 << dst_allocation_view_id);

                    /* partite and its view are already coherent on that device */
                    MemoryReplica & dst_replica = partite.block->replicas[device_global_id];
                    if (dst_replica.coherency & dst_allocbit)
                    {
                        partite.must_fetch = false;
                        continue ;
                    }

                    /* retrieve an incoherent allocation view */
                    MemoryReplicaAllocationView * dst_allocation_view = dst_replica.allocations[dst_allocation_view_id];

                    /* increment task fetch counter */
                    LOGGER_DEBUG(
                        "task `%s` fetching by `%s` on `%p`",
                        access->task->label,
                        (dst_replica.fetching & dst_allocbit) ? "awaiting" : (partite.block->fetching) ? "forwarding" : "launching",
                        (void *) dst_allocation_view->view.addr
                    );

                    /* partite is already being fetched on that device */
                    if (dst_replica.fetching & dst_allocbit)
                    {
                        partite.must_fetch = false;
                        LOGGER_DEBUG("Skipping fetch of a block already being fetched (concurrent read)");

                        /* register a task awaiting on the fetch completion */
                        __task_fetching(1, access->task);
                        dst_allocation_view->awaiting.accesses.push_back(access);

                        continue ;
                    }

                    ///////////////
                    // SETUP SRC //
                    ///////////////

                    // find source:
                    //  - if its already coherent on a device, use it as a source
                    //  - else, if its already transfering from the host to any device, wait for it and forward using D2D (PCI contention heuristic)
                    //  - else, transfer H2D

                    if (partite.block->coherency & ~(1 << HOST_DEVICE_GLOBAL_ID))
                    {
                        partite.must_fetch = true;

                        /* create dst view */
                        partite.dst_device_global_id = device_global_id;
                        partite.dst_view = dst_allocation_view->view;

                        /* get a coherent source */
                        device_global_id_t src = this->runtime->router.get_source(device_global_id, partite.block->coherency);
                        assert(partite.block->coherency & (1 << src));

                        /* Get the first coherent allocation on that device */
                        MemoryReplica & src_replica = partite.block->replicas[src];
                        assert(src_replica.nallocations > 0);
                        assert(src_replica.coherency);

                        # pragma message(TODO "Instead of getting the first coherent, maybe get the LARGEST coherent, so we maximize odds to merge with future allocations")
                        memory_allocation_view_id_t src_allocation_view_id = (memory_allocation_view_id_t) (__builtin_ffs(src_replica.coherency) - 1);
                        assert(src_replica.coherency & (1 << src_allocation_view_id));
                        assert(0 <= src_allocation_view_id && src_allocation_view_id < src_replica.nallocations);

                        /* Retrieve and set src view infos */
                        MemoryReplicaAllocationView * src_allocation_view = src_replica.allocations[src_allocation_view_id];
                        partite.src_device_global_id    = src;
                        partite.src_allocation_view_id  = src_allocation_view_id;
                        partite.src_view                = src_allocation_view->view;
                        partite.src_chunk               = src_allocation_view->chunk;
                    }
                    # if USE_D2D_FORWARDING
                    /* heuristic: if another device is already fetching from the host, register a forward callback instead to reduce PCI contention */
                    else if (partite.block->fetching & ~(1 << HOST_DEVICE_GLOBAL_ID))
                    {
                        /* the fetch will be initiated by the other device that
                         * is already fetching that data for a D2D transfer.
                         * No need to create partite.src and partite.dst as we
                         * are not fetching now */
                        partite.must_fetch = false;

                        /* one device is already fetching, add a D2D forward callback */
                        device_global_id_t fetching_device_global_id = this->runtime->router.get_source(device_global_id, partite.block->fetching);
                        assert(0 <= fetching_device_global_id && fetching_device_global_id < XKRT_DEVICES_MAX);
                        assert(partite.block->fetching & (1 << fetching_device_global_id));

                        MemoryReplica & fetching_replica = partite.block->replicas[fetching_device_global_id];
                        assert(fetching_replica.fetching);

                        // Maybe there is several fetching alloc ? in such case, which one to pick ?
                        // Currently select the first one
                        memory_allocation_view_id_t fetching_allocation_view_id = (memory_allocation_view_id_t) (__builtin_ffs(fetching_replica.fetching) - 1);
                        assert(0 <= fetching_allocation_view_id && fetching_allocation_view_id < fetching_replica.nallocations);
                        assert(fetching_replica.fetching & (1 << fetching_allocation_view_id));

                        MemoryReplicaAllocationView * fetching_allocation_view = fetching_replica.allocations[fetching_allocation_view_id];
                        assert(fetching_allocation_view);

                        LOGGER_DEBUG("registered a forward for task `%s`", access->task->label);

                        const MemoryForward forward(access, partition.chunk, partite.hyperrect, device_global_id, dst_allocation_view->view);
                        fetching_allocation_view->awaiting.forwards.push_back(forward);
                        __task_fetching(1, access->task);
                    }
                    # endif /* USE_D2D_FORWARDING */
                    else
                    {
                        assert(!partite.block->coherency);
                        # if USE_D2D_FORWARDING
                        assert(!partite.block->fetching);
                        # endif /* !USE_D2D_FORWARDING */

                        partite.must_fetch = true;

                        /* create dst view */
                        partite.dst_device_global_id = device_global_id;
                        partite.dst_view = dst_allocation_view->view;

                        /* using host as src, which is assumed coherent */
                        partite.src_device_global_id    = HOST_DEVICE_GLOBAL_ID;
                        partite.src_allocation_view_id  = MEMORY_REPLICATE_ALLOCATION_VIEW_NONE;
                    }

                    /* update bitfields so no other concurrent fetch occurs */
                    dst_replica.fetching  |= dst_allocbit;
                    partite.block->fetching |= dst_devbit;

                    //////////////////////////////
                    // ASSERTION ON SRC AND DST //
                    //////////////////////////////

                    assert(partite.dst_device_global_id   != partite.src_device_global_id ||
                           partite.dst_allocation_view_id != partite.src_allocation_view_id);
                }
            }
        }

        inline void
        fetch_access_set_coherent(
            access_t * access,
            device_global_id_t device_global_id,
            Partition & partition
        ) {
            assert(this->is_locked());

            /* if access has a write mode, make all copies incoherent */
            if (access->mode & ACCESS_MODE_W)
            {
                const device_global_id_bitfield_t devbit = (device_global_id_bitfield_t) (1 << device_global_id);
                for (Partite & partite : partition.partites)
                {
                    const memory_allocation_view_id_bitfield_t allocbit = (memory_allocation_view_id_bitfield_t) (1 << partite.dst_allocation_view_id);

                    /* make all replicas incoherent */
                    # pragma message(TODO "Can coherency be managed in a lazier way ?")
                    for (device_global_id_t device_global_id = 0 ;
                            device_global_id < XKRT_DEVICES_MAX ;
                            ++device_global_id)
                    {
                        MemoryReplica & replica = partite.block->replicas[device_global_id];
                        replica.coherency = 0;
                    }
                    partite.block->coherency = 0;

                    /* There is no concurrent access anyway, so make memory coherent now
                     * (even though the kernel has not executed, and the data is not rigourously coherent yet) */
                    MemoryReplica & replica = partite.block->replicas[device_global_id];
                    replica.coherency = allocbit;
                    partite.block->coherency = devbit;
                }
            }
        }

        /* launch a single fetch */
        inline void
        fetch_list_launch_ith(
            access_t * access,
            fetch_list_t * list,
            task_wait_counter_type_t i
        ) {
            assert(access);
            assert(access->task);
            assert(i >= 0);
            assert(i < list->n);
            assert(i < list->capacity);

            /* retrieve the fetch, and cancel if it got merged */
            fetch_t * fetch = list->fetches + i;
            if (fetch->merged)
                return ;

            /* callback setup */
            assert(XKRT_CALLBACK_ARGS_MAX >= 4);
            callback_t callback;
            callback.func = fetch_callback;
            callback.args[0] = this->runtime;
            callback.args[1] = access;
            callback.args[2] = list;
            callback.args[3] = (void *) i;

            /* the device on which a queue will perform the device - use the dst device if not the host */
            assert(fetch->src_device_global_id != HOST_DEVICE_GLOBAL_ID || fetch->dst_device_global_id != HOST_DEVICE_GLOBAL_ID);
            device_global_id_t device_global_id = (fetch->dst_device_global_id != HOST_DEVICE_GLOBAL_ID) ? fetch->dst_device_global_id : fetch->src_device_global_id;

            /* launch asynchronous copy */
            if (access->type == ACCESS_TYPE_SEGMENT)
            {
                assert(fetch->host_view.n == 1);
                assert(fetch->host_view.sizeof_type == 1);
                this->runtime->copy(
                    device_global_id,
                    (size_t) fetch->host_view.m,
                    fetch->dst_device_global_id,
                    (uintptr_t) fetch->dst_view.addr,
                    fetch->src_device_global_id,
                    (uintptr_t) fetch->src_view.addr,
                    callback
                );
            }
            else
            {
                this->runtime->copy(
                    device_global_id,
                    fetch->host_view,
                    fetch->dst_device_global_id,
                    fetch->dst_view,
                    fetch->src_device_global_id,
                    fetch->src_view,
                    callback
                );
            }
        }

        inline void
        fetch_list_launch(
            access_t * access,
            fetch_list_t * list
        ) {
            for (task_wait_counter_type_t i = 0 ; i < list->n ; ++i)
                fetch_list_launch_ith(access, list, i);
        }

        template <bool only_allocates = false>
        inline fetch_list_t *
        fetch_list_to_device(
            access_t * access,
            device_global_id_t device_global_id
        ) {
            assert(access->type == ACCESS_TYPE_SEGMENT ||
                    access->type == ACCESS_TYPE_BLAS_MATRIX);

            // run the coherency protocol
            Search search(device_global_id);
            fetch_list_t * list = NULL;

            this->lock();
            {
                # pragma message(TODO "Step (1) and (2) could be merged to only search once")

                # if 1
                LOGGER_DEBUG(
                    "access_t<2>(MATRIX_COLMAJOR, (void *) %p, %lu, %lu, %lu, %lu, %s),",
                    (void *) access->host_view.addr,
                    access->host_view.ld,
                    access->host_view.m,
                    access->host_view.n,
                    access->host_view.sizeof_type,
                    xkrt_access_mode_to_str(access->mode)
                );
                # endif

                /* step (1) ensure the access is represented in the tree as a partition of rects */
                search.prepare_insert(access);
                for (Rect & rect : access->rects())
                    this->insert(search, rect);

                /* step (2) find all rects representing the access */
                search.prepare_search_partition();
                for (const Rect & rect : access->rects())
                    this->intersect(search, rect);
                assert(search.partition.partites.size() >= 1);

                /* step (3) find or allocate a contiguous memory view for that access on that device */
                this->fetch_access_find_allocation(access, device_global_id, search.partition);

                /* step (4) set the access view on the device (that will be used by the kernel) */
                this->fetch_access_setup_replicas(access, device_global_id, search);

                if (!only_allocates)
                {
                    /* step (5) if read access, find src/dst, and setup views to transfer on step (7) */
                    this->fetch_access_setup_copies(access, device_global_id, search.partition);

                    /* step (6) if write access, make all other replicas incoherent */
                    this->fetch_access_set_coherent(access, device_global_id, search.partition);
                }

            } /* this->lock(); */
            this->unlock();

            /* step (7) - convert a partition to the minimum number of fetches to run */
            if (!only_allocates)
                if (access->mode & ACCESS_MODE_R)
                    list = this->fetch_list_from_partition(search.partition);

            return list;
        }

        /** Fetch the access on the given device */
        void
        fetch(
            access_t * access,
            device_global_id_t device_global_id
        ) {
            if (access->state == ACCESS_STATE_FETCHING || access->state == ACCESS_STATE_FETCHED)
                return ;

            assert(access->state == ACCESS_STATE_INIT);
            access->state = ACCESS_STATE_FETCHING;

            LOGGER_DEBUG("Fetching an access for task `%s`",
                    access->task ? access->task->label : "(null)");

            // no need to fetch unified memory
            if (access->scope == ACCESS_SCOPE_UNIFIED)
            {
                // TODO: cuda does not provide a 'cuMemAdvise2D' so kinda fucked here
                # if 1
                this->runtime->memory_unified_advise(
                    device_global_id,
                    (const void *) access->host_view.begin_addr(),
                    (size_t) (access->host_view.end_addr() - access->host_view.begin_addr())
                );
                # endif

                # if 1
                // TODO: cuda does not provide a 'cuMemAdvise2D' so kinda fucked here
                this->runtime->memory_unified_prefetch(
                    device_global_id,
                    (const void *) access->host_view.begin_addr(),
                    (size_t) (access->host_view.end_addr() - access->host_view.begin_addr())
                );
                # endif

                // set host addr/size
                access->device_view.addr = access->host_view.begin_addr();
                access->device_view.ld   = access->host_view.ld;
                access->state            = ACCESS_STATE_FETCHED;
                return ;
            }

            fetch_list_t * list;

            // short-path if targetting the host
            if (device_global_id == HOST_DEVICE_GLOBAL_ID)
            {
                list = this->fetch_list_to_host(access);
            }
            // long-path if targetting a device
            else
            {
                list = this->fetch_list_to_device(access, device_global_id);
            }

            // if there is fetch to perform, launch them
            if (list)
            {
                // reduce them
                if (this->merge_transfers)
                {
                    LOGGER_FATAL("Transfer merge not supported");
                    // this->fetch_list_reduce(list);
                }

                // increase task wait counter
                __task_fetching(list->pending, access->task);
                this->fetch_list_launch(access, list);
            }
        }

        ////////////////////////
        // ALLOCATE TO DEVICE //
        ////////////////////////
        void
        allocate_to_device(
            access_t * access,
            device_global_id_t device_global_id
        ) {
            assert(access->task == NULL);
            assert(device_global_id != HOST_DEVICE_GLOBAL_ID);
            this->fetch_list_to_device<true>(access, device_global_id);
        }

        //////////////////
        //  INVALIDATE  //
        //////////////////
        void
        invalidate(void)
        {
            // empty the tree
            this->clear();
        }

        //////////////
        //  OCR     //
        //////////////

        device_global_id_bitfield_t
        who_owns(access_t * access)
        {
            // find how much bytes are owned per device
            Search search;
            search.prepare_search_owners();
            this->lock();
            {
                for (const Rect & rect : access->rects())
                    if (!rect.is_empty())
                        this->intersect(search, rect);
            }
            this->unlock();

            // find devices which owns the most bytes
            device_global_id_bitfield_t owners = 0;
            size_t bytes_owned_max = 0;
            for (device_global_id_t device_global_id = 0 ; device_global_id < XKRT_DEVICES_MAX ; ++device_global_id)
            {
                const size_t bytes_owned = search.bytes_owned[device_global_id];
                if (bytes_owned_max < bytes_owned)
                {
                    bytes_owned_max = bytes_owned;
                    owners = (device_global_id_bitfield_t) (1 << device_global_id);
                }
                else if (bytes_owned_max && bytes_owned_max == bytes_owned)
                    owners |= (device_global_id_bitfield_t) (1 << device_global_id);
            }

            return owners;
        }

        //////////////
        //  INSERT  //
        //////////////

        void
        on_insert(
            NodeBase * nodebase,
            Search & search
        ) {
            (void) nodebase;
            (void) search;
            switch (search.type)
            {
                case (Search::Type::INSERTING_BLOCKS):
                    break ;

                # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
                case (Search::Type::REGISTER):
                {
                    Node * node = reinterpret_cast<Node *>(nodebase);
                    node->block.registered = true;
                    break ;
                }

                case (Search::Type::UNREGISTER):
                {
                    Node * node = reinterpret_cast<Node *>(nodebase);
                    node->block.registered = false;
                    break ;
                }
                # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

                default:
                    LOGGER_FATAL("Invalid search type for insert");
            }
        }

        /* shrinking on dimension 'k' from 'this->hyperrect[k]' to 'interval' */
        void
        on_shrink(
            NodeBase * nodebase,
            const Interval & interval,
            int k
        ) {
            static_assert(K == 2);
            Node * node = reinterpret_cast<Node *>(nodebase);

            assert(k < K);
            assert(node->hyperrect[k].includes(interval));

            ///////////////////////
            //  SHRINK HOST VIEW //
            ///////////////////////

            assert(node->hyperrect[k].a <= interval.a);
            const INTERVAL_DIFF_TYPE_T da = interval.a - node->hyperrect[k].a;

            assert(node->hyperrect[k].b >= interval.b);

            // must be aligned on sizeof(type)
            if (k == ACCESS_BLAS_ROW_DIM)
            {
                const INTERVAL_DIFF_TYPE_T db = node->hyperrect[k].b - interval.b;
                (void) db;
                assert(da % this->sizeof_type == 0);
                assert(db % this->sizeof_type == 0);
            }

            // shrinked-left, gotta offset the views
            if (da)
            {
                // REPLICATES VIEW
                for (MemoryReplica & replica : node->block.replicas)
                {
                    for (memory_allocation_view_id_t i = 0 ; i < replica.nallocations ; ++i)
                    {
                        MemoryReplicaAllocationView * allocation_view = replica.allocations[i];
                        const INTERVAL_DIFF_TYPE_T offset = (k == ACCESS_BLAS_ROW_DIM) ? da : (da * allocation_view->view.ld * this->sizeof_type);
                        allocation_view->view.addr += offset;
                        assert(allocation_view->view.addr >= allocation_view->chunk->ptr);
                    }
                }
            }
        }

        //////////////////
        //  INTERSECT   //
        //////////////////
        inline bool
        intersect_stop_test(
            NodeBase * nodebase,
            Search & search,
            const Rect & h
        ) const {

            (void) nodebase;
            (void) search;
            (void) h;

            // TODO : can we fasten intersection by keeping track of an included 'coherency' bitmask ?

            return false;
        }

        /**
         * The passed rect is intersecting with 'this'
         */
        inline void
        on_intersect(
            NodeBase * nodebase,
            Search & search,
            const Rect & h
        ) const {

            assert(nodebase);
            Node * node = reinterpret_cast<Node *>(nodebase);
            assert(h.intersects(node->hyperrect));

            switch (search.type)
            {
                case (Search::Type::SEARCH_FOR_PARTITION):
                {
                    /* intersecting against 'rect' that had been inserted
                     * previously, so 'node' must be a sub-block of 'rect' */
                    assert(h.includes(node->hyperrect));
                    search.partition.partites.push_back(Partite(&(node->block), node->hyperrect));
                    break ;
                }

                /* search after an access got fetched, to unset 'fetching' bits
                 * and search for tasks awaiting on that rect for a given
                 * allocation */
                case (Search::Type::SEARCH_FETCHED):
                {
                    const device_global_id_bitfield_t devbit = (device_global_id_bitfield_t) (1 << search.device_global_id);
                    MemoryReplica & replica = node->block.replicas[search.device_global_id];

                    if (search.device_global_id != HOST_DEVICE_GLOBAL_ID)
                    {
                        /* this is called after completing a fetch.
                         * There must be at least one allocation available (the one that just got fetched...)
                         */
                        assert(replica.nallocations);

                        /* for each allocation of that block */
                        for (memory_allocation_view_id_t allocation_view_id = 0 ; allocation_view_id < replica.nallocations ; ++allocation_view_id)
                        {
                            const memory_allocation_view_id_bitfield_t allocbit = (memory_allocation_view_id_bitfield_t) (1 << allocation_view_id);
                            MemoryReplicaAllocationView * allocation_view = replica.allocations[allocation_view_id];

                            /* if it matches the allocation being searched */
                            if (allocation_view->chunk == search.chunk)
                            {
                                /* move the awaiting tasks */
                                search.awaiting.accesses.insert(
                                    search.awaiting.accesses.end(),
                                    allocation_view->awaiting.accesses.begin(),
                                    allocation_view->awaiting.accesses.end()
                                );
                                allocation_view->awaiting.accesses.clear();

                                /* move awaiting forwards */
                                search.awaiting.forwards.insert(
                                    search.awaiting.forwards.end(),
                                    allocation_view->awaiting.forwards.begin(),
                                    allocation_view->awaiting.forwards.end()
                                );
                                allocation_view->awaiting.forwards.clear();

                                /* this replica just got fetched and is now coherent */

                                // this assertion is not always true, if coming from
                                // an ACCESS_MODE_W, the data was already set coherent
                                // assert((replica.coherency & allocbit) == 0);
                                replica.coherency |= (memory_allocation_view_id_bitfield_t) allocbit;

                                assert(replica.fetching & allocbit);
                                replica.fetching &= (memory_allocation_view_id_bitfield_t) ~allocbit;

                                break ;
                            }
                        }
                        // at least one allocation must match with the chunk on which we fetched
                        assert(replica.coherency);
                    }

                    /* set device bits */
                    node->block.coherency |= devbit;
                    if (replica.fetching == 0)
                        node->block.fetching &= ~devbit;

                    break ;
                }

                /* search for owners of the access */
                case (Search::Type::SEARCH_OWNERS):
                {
                    Rect intersect;
                    Rect::intersection(&intersect, h, node->hyperrect);
                    const size_t bytes = intersect.size();
                    for (device_global_id_t device_global_id = 0 ; device_global_id < XKRT_DEVICES_MAX ; ++device_global_id)
                        if (node->block.coherency & (1 << device_global_id))
                            search.bytes_owned[device_global_id] += bytes;
                    break ;
                }

                default:
                {
                    LOGGER_FATAL("Invalid search type in memory tree");
                    assert(0);
                }
            }
        }

        Node *
        new_node(
            Search & search,
            const Rect & h,
            const int k,
            const Color color
        ) const {
            assert(
                search.type == Search::Type::INSERTING_BLOCKS
                # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
                || search.type == Search::Type::REGISTER
                # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */
            );
            return new Node(search.access, h, k, color);
        }

        Node *
        new_node(
            Search & search,
            const Rect & h,
            const int k,
            const Color color,
            const NodeBase * inherit
        ) const {
            (void) search;
            assert(
                search.type == Search::Type::INSERTING_BLOCKS
                # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
                || search.type == Search::Type::REGISTER
                # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */
            );
            assert(!h.intersects(inherit->hyperrect));
            return new Node(h, k, color, reinterpret_cast<const Node *>(inherit), this->sizeof_type);
        }

        # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION

        //////////////////////////////////////////
        // Memory registration / unregistration //
        //////////////////////////////////////////

        /* the given memory segment got (un)registered */
        template<Search::Type T>
        inline void
        registered_update(
            uintptr_t ptr,
            size_t size
        ) {
            const uintptr_t   p = (const uintptr_t) ptr;
            const uintptr_t  pp = p + size;

            const uintptr_t a = p - (p % pagesize);
            const uintptr_t b = pp + (pagesize - (pp % pagesize)) % pagesize;

            assert(a % pagesize == 0);
            assert(b % pagesize == 0);
            assert(a        <= p);
            assert(p + size <= b);
            assert(a < b);
            assert(b - a >= size);

            Rect rects[3];
            interval_to_rects(a, b-a, this->ld, this->sizeof_type, rects);

            /* insert blocks in the tree with the registered bit */
            Search search;
            search.prepare(T);
            this->lock();
            {
                for (Rect & rect : rects)
                    this->insert(search, rect);
            }
            this->unlock();
        }

        void
        registered(
            uintptr_t ptr,
            size_t size
        ) {
            registered_update<Search::Type::REGISTER>(ptr, size);
        }

        void
        unregistered(
            uintptr_t ptr,
            size_t size
        ) {
            registered_update<Search::Type::UNREGISTER>(ptr, size);
        }

        # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

};

using BLASMemoryTree = KBLASMemoryTree<2>;

XKRT_NAMESPACE_END

#endif /* __MEMORY_TREE_HPP__ */
