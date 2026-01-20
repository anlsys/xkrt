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

#ifndef __XKRT_RUNTIME_H__
# define __XKRT_RUNTIME_H__

# include <xkrt/support.h>
# include <xkrt/types.h>
# include <xkrt/conf/conf.h>
# include <xkrt/distribution/distribution.h>
# include <xkrt/driver/driver.h>
# include <xkrt/thread/team.h>
# include <xkrt/thread/thread.h>
# include <xkrt/memory/access/coherency-controller.hpp>
# include <xkrt/memory/access/common/interval-set.hpp>
# include <xkrt/memory/register.h>
# include <xkrt/memory/routing/router-affinity.hpp>
# include <xkrt/stats/stats.h>
# include <xkrt/sync/spinlock.h>
# include <xkrt/task/task.hpp>

# include <hwloc.h>

# include <map>

XKRT_NAMESPACE_BEGIN

/**
 * @brief Main runtime system structure
 *
 * This structure contains all the state and configuration for the XKRT runtime system,
 * including device drivers, task formats, memory management, and threading.
 */
struct  runtime_t
{
    /**
     * @brief Runtime system state enumeration
     */
    enum state_t : uint8_t
    {
        DEINITIALIZED = 0,  ///< Runtime is not initialized
        INITIALIZED   = 1   ///< Runtime is initialized and ready
    };
    std::atomic<state_t> state;  ///< Current runtime state (atomic)

    drivers_t drivers;  ///< List of registered device drivers

    /**
     * @brief Task format registry
     */
    struct {
        task_formats_t list;                       ///< Complete list of task formats
        task_format_id_t host_capture;             ///< Host capture task format ID
        task_format_id_t memory_copy_async;        ///< Async memory copy task format ID
        task_format_id_t memory_touch_async;       ///< Async memory touch task format ID
        task_format_id_t memory_register_async;    ///< Async memory registration format ID
        task_format_id_t memory_unregister_async;  ///< Async memory unregistration format ID
        task_format_id_t file_read_async;          ///< Async file read task format ID
        task_format_id_t file_write_async;         ///< Async file write task format ID
    } formats;

    conf_t conf;  ///< User configuration

    RouterAffinity router;  ///< Memory router used by the MCC

    # if XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION
    std::map<uintptr_t, size_t> registered_memory;  ///< Map of registered memory segments (addr -> size)
    # endif /* XKRT_MEMORY_REGISTER_OVERFLOW_PROTECTION */

    # if XKRT_MEMORY_REGISTER_ASSISTED
    IntervalSet<uintptr_t> registered_pages;  ///< Set of registered memory page intervals
    # endif /* XKRT_MEMORY_REGISTER_ASSISTED */

    hwloc_topology_t topology;  ///< Hardware locality topology (read-only, initialized at startup)

    //////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC INTERFACES //
    //////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////
    // Management //
    ////////////////

    /**
     * @brief Initialize the runtime system
     *
     * Loads devices, drivers and creates task formats
     *
     * @return 0 on success, non-zero on error
     */
    int init(void);

    /**
     * @brief Deinitialize the runtime system
     *
     * Cleans up all resources, destroys drivers and task formats
     *
     * @return 0 on success, non-zero on error
     */
    int deinit(void);

    /**
     * @brief Reset the runtime
     *
     * Deallocates all memory replicas and tasks
     */
    void reset(void);

    void reset_coherence_controllers(void);
    void reset_dependence_controllers(void);

    ////////////////////
    // DATA MOVEMENTS //
    ////////////////////

    /**
     * @brief Submit a 1D copy command to a device queue
     *
     * @param device_global_id Global ID of the device executing the copy
     * @param size Number of bytes to copy
     * @param dst_device_global_id Destination device global ID
     * @param dst_device_addr Destination device address
     * @param src_device_global_id Source device global ID
     * @param src_device_addr Source device address
     * @param callback Callback function to invoke upon command completion
     */
    void copy(
        const device_global_id_t   device_global_id,
        const size_t               size,
        const device_global_id_t   dst_device_global_id,
        const uintptr_t            dst_device_addr,
        const device_global_id_t   src_device_global_id,
        const uintptr_t            src_device_addr,
        const callback_t         & callback
    );

    /**
     * @brief Submit a 2D copy command to a device queue
     *
     * @param device_global_id Global ID of the device executing the copy
     * @param host_view Host memory view
     * @param dst_device_global_id Destination device global ID
     * @param dst_device_view Destination device memory replica view
     * @param src_device_global_id Source device global ID
     * @param src_device_view Source device memory replica view
     * @param callback Callback function to invoke upon completion
     */
    void copy(
        const device_global_id_t      device_global_id,
        const memory_view_t         & host_view,
        const device_global_id_t      dst_device_global_id,
        const memory_replica_view_t & dst_device_view,
        const device_global_id_t      src_device_global_id,
        const memory_replica_view_t & src_device_view,
        const callback_t            & callback
    );

    /**
     * @brief Asynchronously copy memory by spawning n parallel tasks
     *
     * Spawns 'n' tasks where each task i in [0..n-1]:
     *   - oi = i*size/n (offset)
     *   - ai = ptr + oi (start address)
     *   - bi = MIN(ai+size/n, size) (end address)
     *   - si = bi - ai (chunk size)
     *   - Accesses: virtual read on Interval(src.ai, src.bi)
     *   - Routine: submit a copy command for this chunk
     *
     * @param device_global_id Global ID of the device executing the copy
     * @param size Total number of bytes to copy
     * @param dst_device_global_id Destination device global ID
     * @param dst_device_addr Destination device base address
     * @param src_device_global_id Source device global ID
     * @param src_device_addr Source device base address
     * @param n Number of parallel tasks to spawn
     */
    void memory_copy_async(
        const device_global_id_t   device_global_id,
        const size_t               size,
        const device_global_id_t   dst_device_global_id,
        const uintptr_t            dst_device_addr,
        const device_global_id_t   src_device_global_id,
        const uintptr_t            src_device_addr,
        int                        n
    );

    ///////////////////////
    // DATA DISTRIBUTION //
    ///////////////////////

    /**
     * @brief Asynchronously distribute memory uniformly across all devices, by
     * spawning empty tasks that reads a sub segment
     *
     * @param type Distribution type (e.g., block, cyclic)
     * @param ptr Pointer to the memory segment
     * @param size Size of the memory segment
     * @param nb Block size for distribution
     * @param h Halo/overlap size
     */
    void distribute_async(
        distribution_type_t type,
        void * ptr, size_t size,
        size_t nb,
        size_t h
    );

    /**
     * @brief Asynchronously distribute array of segments across all devices, by spawning 'n' empty tasks that read each segment
     *
     * @param type Distribution type
     * @param ptr Array of pointers to memory segments
     * @param chunk_size Size of each chunk
     * @param n Number of chunks
     */
    void distribute_async(
        distribution_type_t type,
        void ** ptr,
        size_t chunk_size,
        unsigned int n
    );

    /**
     * @brief Asynchronously distribute a matrix across all devices
     *
     * @param type Distribution type (e.g., 2D block-cyclic)
     * @param storage Matrix storage layout (row-major, column-major)
     * @param ptr Pointer to matrix data
     * @param ld Leading dimension of the matrix
     * @param m Number of rows
     * @param n Number of columns
     * @param mb Row block size
     * @param nb Column block size
     * @param sizeof_type Size of each matrix element in bytes
     * @param hx Horizontal halo/overlap size
     * @param hy Vertical halo/overlap size
     */
    void distribute_async(
        distribution_type_t type,
        matrix_storage_t storage,
        void * ptr, size_t ld,
        size_t m, size_t n,
        size_t mb, size_t nb,
        size_t sizeof_type,
        size_t hx, size_t hy
    );

    /////////////////////
    // I/O FILESYSTEM //
    /////////////////////

    /**
     * @brief Asynchronously read from file using parallel tasks
     *
     * Spawns 'n' tasks where each task i in [0..n-1]:
     *   - oi = i*size/n (file offset)
     *   - ai = ptr + oi (buffer address)
     *   - bi = MIN(ai+size/n, size)
     *   - si = bi - ai (bytes to read)
     *   - Access: write on Interval(ai, bi)
     *   - Routine: fseek to 'oi' and read 'si' bytes to [ai..bi]
     *
     * @param fd File descriptor
     * @param buffer Destination buffer
     * @param size Total bytes to read
     * @param n Number of parallel read tasks
     * @return 0 on success, non-zero on error
     */
    int file_read_async (int fd, void * buffer, const size_t size, int n);

    /**
     * @brief Asynchronously write to file using parallel tasks
     *
     * Spawns 'n' tasks where each task i in [0..n-1]:
     *   - oi = i*size/n (file offset)
     *   - ai = ptr + oi (buffer address)
     *   - bi = MIN(ai+size/n, size)
     *   - si = bi - ai (bytes to write)
     *   - Access: read on Interval(ai, bi)
     *   - Routine: fseek to 'oi' and write 'si' bytes from [ai..bi]
     *
     * @param fd File descriptor
     * @param buffer Source buffer
     * @param size Total bytes to write
     * @param n Number of parallel write tasks
     * @return 0 on success, non-zero on error
     */
    int file_write_async(int fd, void * buffer, const size_t size, int n);

    /**
     * @brief Iterate over memory segment chunks and apply a function
     *
     * Divides the memory segment [p, p+size) into n chunks and calls
     * func(i, a, b) for each chunk where i is the chunk index and
     * [a, b) is the address range.
     *
     * @param p Base address of memory segment
     * @param size Total size in bytes
     * @param n Number of chunks
     * @param func Function to call for each chunk: func(index, start_addr, end_addr)
     */
    inline void foreach(
        const uintptr_t p,
        const size_t size,
        int n,
        const std::function<void(const int, const uintptr_t, const uintptr_t)> & func)
    {
        // compute number of commands to spawn
        if (size < (size_t) n)
            n = (int) size;

        // compute chunk size
        const size_t chunksize = size / n;
        assert(chunksize > 0);

        for (int i = 0; i < n; ++i) {
            const uintptr_t a = p + i * chunksize;
            const uintptr_t b = (i == n - 1) ? p + size : p + (i + 1) * chunksize;
            func(i, a, b);
        }
    }

    ///////////////////////
    // MEMORY ALLOCATION //
    ///////////////////////

    /**
     * @brief Ensure chunk0 is preallocated on the device
     *
     * @param device_global_id Global device identifier
     * @param memory_id Device memory index
     */
    void memory_device_preallocate_ensure(const device_global_id_t device_global_id, const int memory_id);

    /**
     * @brief Allocate memory on a specific device memory bank
     *
     * @param device_global_id Global device identifier
     * @param size Size in bytes to allocate
     * @param memory_id Device memory index (bank)
     * @return Pointer to allocated chunk, or nullptr on failure
     */
    area_chunk_t * memory_device_allocate_on(const device_global_id_t device_global_id, const size_t size, const int memory_id);

    /**
     * @brief Allocate memory on device's default memory bank
     *
     * @param device_global_id Global device identifier
     * @param size Size in bytes to allocate
     * @return Pointer to allocated chunk, or nullptr on failure
     */
    area_chunk_t * memory_device_allocate(const device_global_id_t device_global_id, const size_t size);

    /**
     * @brief Deallocate a device memory chunk
     *
     * @param device_global_id Global device identifier
     * @param chunk Pointer to chunk to deallocate
     */
    void memory_device_deallocate(const device_global_id_t device_global_id, area_chunk_t * chunk);

    /**
     * @brief Deallocate all memory previously allocated on the device
     *
     * @param device_global_id Global device identifier
     */
    void memory_device_deallocate_all(const device_global_id_t device_global_id);

    /**
     * @brief Allocate host memory using a device driver
     *
     * @param device_global_id Global device identifier (determines which driver to use)
     * @param size Size in bytes to allocate
     * @return Pointer to allocated memory, or nullptr on failure
     */
    void * memory_host_allocate(const device_global_id_t device_global_id, const size_t size);

    /**
     * @brief Deallocate host memory using a device driver
     *
     * @param device_global_id Global device identifier (determines which driver to use)
     * @param mem Pointer to memory to deallocate
     * @param size Size in bytes
     */
    void memory_host_deallocate(const device_global_id_t device_global_id, void * mem, const size_t size);

    /**
     * @brief Allocate unified memory accessible by both host and device
     *
     * @param device_global_id Global device identifier (determines which driver to use)
     * @param size Size in bytes to allocate
     * @return Pointer to allocated unified memory, or nullptr on failure
     */
    void * memory_unified_allocate(const device_global_id_t device_global_id, const size_t size);

    /**
     * @brief Deallocate unified memory
     *
     * @param device_global_id Global device identifier (determines which driver to use)
     * @param mem Pointer to unified memory to deallocate
     * @param size Size in bytes
     */
    void memory_unified_deallocate(const device_global_id_t device_global_id, void * mem, const size_t size);

    /////////////////////
    // MEMORY MOVEMENT //
    /////////////////////

    /**
     * @brief Synchronously allocate a non-coherent device replica
     *
     * @param device_global_id Global device identifier
     * @param ptr Host memory pointer
     * @param size Size in bytes
     */
    void memory_noncoherent_alloc(device_global_id_t device_global_id, void * ptr, size_t size);

    /**
     * @brief Synchronously allocate a non-coherent device replica for a matrix
     *
     * @param device_global_id Global device identifier
     * @param storage Matrix storage layout
     * @param ptr Host matrix pointer
     * @param ld Leading dimension
     * @param m Number of rows
     * @param n Number of columns
     * @param sizeof_type Size of each element in bytes
     */
    void memory_noncoherent_alloc(device_global_id_t device_global_id, matrix_storage_t storage, void * ptr, size_t ld, size_t m, size_t n, size_t sizeof_type);

    /**
     * @brief Spawn an empty task with a read access on the passed segment
     *
     * @param device_global_id Global device identifier
     * @param ptr Host memory pointer
     * @param size Size in bytes
     */
    void memory_coherent_async(device_global_id_t device_global_id, void * ptr, size_t size);

    /**
     * @brief Spawn an empty task with a read access on the passed matrix
     *
     * @param device_global_id Global device identifier
     * @param storage Matrix storage layout
     * @param ptr Host matrix pointer
     * @param ld Leading dimension
     * @param m Number of rows
     * @param n Number of columns
     * @param sizeof_type Size of each element in bytes
     */
    void memory_coherent_async(device_global_id_t device_global_id, matrix_storage_t storage, void * ptr, size_t ld, size_t m, size_t n, size_t sizeof_type);

    /**
     * @brief Spawn an empty undeferred task with a read access on the passed segment
     *
     * @param device_global_id Global device identifier
     * @param ptr Host memory pointer
     * @param size Size in bytes
     */
    void memory_coherent_sync(device_global_id_t device_global_id, void * ptr, size_t size);

    /**
     * @brief Spawn an empty undeferred task with a read access on the passed matrix
     *
     * @param device_global_id Global device identifier
     * @param storage Matrix storage layout
     * @param ptr Host matrix pointer
     * @param ld Leading dimension
     * @param m Number of rows
     * @param n Number of columns
     * @param sizeof_type Size of each element in bytes
     */
    void memory_coherent_sync(device_global_id_t device_global_id, matrix_storage_t storage, void * ptr, size_t ld, size_t m, size_t n, size_t sizeof_type);

    /**
     * @brief Hint to unified memory system about future device access
     *
     * @param device_global_id Global device identifier
     * @param addr Memory address
     * @param size Size in bytes
     * @return 0 on success, non-zero on error
     */
    int memory_unified_advise  (const device_global_id_t device_global_id, const void * addr, const size_t size);

    /**
     * @brief Prefetch unified memory to make it coherent on device
     *
     * @param device_global_id Global device identifier
     * @param addr Memory address
     * @param size Size in bytes
     * @return 0 on success, non-zero on error
     */
    int memory_unified_prefetch(const device_global_id_t device_global_id, const void * addr, const size_t size);

    /////////////////////////
    // MEMORY REGISTRATION //
    /////////////////////////

    /**
     *  TODO: is it 'legal' to have concurrently
     *      - host thread A writing to memory [a..b]
     *      - host thread B pinning memory [a..b]
     * ? Can't find any doc on it, so for now, assuming it is not
     *
     *  Spawn 'n' tasks so that each task i in [0..n-1]
     *      ai = ptr + i*size/n
     *      bi = MIN(ai+size/n, size)
     *
     *      - access    - write on Interval(ai, bi)
     *                  - virtual commutative write on NULL
     *
     *      - routine - register/unregister/touch [ai..bi]
     */



    /**
     * @brief Synchronously register memory for device access (pinning)
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @return 0 on success, non-zero on error
     */
    int memory_register  (void * ptr, size_t size);

    /**
     * @brief Synchronously unregister memory
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @return 0 on success, non-zero on error
     */
    int memory_unregister(void * ptr, size_t size);

    /**
     * @brief Asynchronously register memory using a single task
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @return 0 on success, non-zero on error
     */
    int memory_register_async  (void * ptr, size_t size);

    /**
     * @brief Asynchronously unregister memory using a single task
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @return 0 on success, non-zero on error
     */
    int memory_unregister_async(void * ptr, size_t size);

    /**
     * @brief Asynchronously register memory using parallel tasks
     *
     * Spawns 'n' tasks where each task i in [0..n-1]:
     *   - ai = ptr + i*size/n
     *   - bi = MIN(ai+size/n, size)
     *   - Access: write on Interval(ai, bi), virtual commutative write on NULL
     *   - Routine: register [ai..bi]
     *
     * @note Assumes it is NOT safe to have concurrent host writes during registration
     *
     * @param team Thread team to use for task execution
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param n Number of parallel tasks
     * @return 0 on success, non-zero on error
     */
    int memory_register_async  (team_t * team, void * ptr, const size_t size, int n);

    /**
     * @brief Asynchronously unregister memory using parallel tasks
     *
     * @param team Thread team to use for task execution
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param n Number of parallel tasks
     * @return 0 on success, non-zero on error
     */
    int memory_unregister_async(team_t * team, void * ptr, const size_t size, int n);

    /**
     * @brief Asynchronously touch memory pages to ensure they're resident
     *
     * @param team Thread team to use for task execution
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param n Number of parallel tasks
     * @return 0 on success, non-zero on error
     */
    int memory_touch_async     (team_t * team, void * ptr, const size_t size, int n);

    /**
     * @brief Asynchronously register memory using default host team
     *
     * Convenience wrapper that uses the host driver's team
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param n Number of parallel tasks
     * @return 0 on success, non-zero on error
     */
    int memory_register_async(void * ptr, const size_t size, int n)
    {
        team_t * team = this->team_get(XKRT_DRIVER_TYPE_HOST, 0);
        assert(team);
        return this->memory_register_async(team, ptr, size, n);
    }

    /**
     * @brief Asynchronously unregister memory using default host team
     *
     * Convenience wrapper that uses the host driver's team
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param n Number of parallel tasks
     * @return 0 on success, non-zero on error
     */
    int memory_unregister_async(void * ptr, const size_t size, int n)
    {
        team_t * team = this->team_get(XKRT_DRIVER_TYPE_HOST, 0);
        assert(team);
        return this->memory_unregister_async(team, ptr, size, n);
    }

    /**
     * @brief Asynchronously touch memory pages to ensure they're resident
     *
     * Convenience wrapper that uses the host driver's team
     *
     * @param ptr Memory pointer
     * @param size Size in bytes
     * @param n Number of parallel tasks
     * @return 0 on success, non-zero on error
     */
    int memory_touch_async     (void * ptr, const size_t size, int n)
    {
        team_t * team = this->team_get(XKRT_DRIVER_TYPE_HOST, 0);
        assert(team);
        return this->memory_touch_async(team, ptr, size, n);
    }

    /////////////
    // TASKING //
    /////////////

    /* Commit a task - so it may be schedule from now once its dependences
     * completed. The task will be pushed to a device team */
    void task_commit(task_t * task);

    /* Decrease the ref counter of a detachable task, and complete it if it reaches 0 */
    void task_detachable_decr(task_t * task);

    /* Increase the ref counter of a detachable task */
    void task_detachable_incr(task_t * task);

    /* Complete a task */
    void task_complete(task_t * task);

    /* wait for children tasks of the current task to complete */
    void task_wait(void);

    /* enqueue a task to :
     *  - the current thread if its within a team
     *  - or the host driver team if the current thread has no team
     */
    void task_enqueue(task_t * task);
    static inline void
    task_enqueue(runtime_t * runtime, task_t * task)
    {
        runtime->task_enqueue(task);
    }

    /* enqueue a task to the given thread */
    void task_thread_enqueue(thread_t * thread, task_t * task);
    static inline void task_thread_enqueue(runtime_t * runtime, thread_t * thread, task_t * task)
    {
        runtime->task_thread_enqueue(thread, task);
    }

    /* enqueue a task to the given team */
    void task_team_enqueue(team_t * team, task_t * task);
    static inline void
    task_team_enqueue(runtime_t * runtime, team_t * team, task_t * task)
    {
        return runtime->task_team_enqueue(team, task);
    }

    /* duplicate a moldable task (do not use unless you know what you're doing) */
    task_t * task_dup(const task_t * task);

    /* prototype of a task routine */
    typedef std::function<void(runtime_t *, device_t *, task_t *)> task_routine_t;

    /* prototype of a routine to set task accesses */
    typedef std::function<void(task_t *, access_t *)> task_accesses_setter_t;

    ////////////////////////////////
    // TASK INSTANCIATION HELPERS //
    ////////////////////////////////

    /* Allocate a new task and call its constructor */
    template <bool use_thread_stack = true>
    inline task_t * task_new(const task_format_id_t fmtid, const task_flag_bitfield_t flags, const size_t size)
    {
        task_t * task = this->task_allocate<use_thread_stack>(size);
        new (task) task_t(fmtid, flags);
        return task;
    }

    /* Allocate a new task */
    template <bool use_thread_stack>
    task_t * task_allocate(const size_t size);

    /* Deallocate all tasks previously allocated by the calling thread */
    void task_deallocate_all(void);

    template <task_flag_bitfield_t flags>
    inline task_t *
    task_instanciate(
        const task_format_id_t fmtid,
        const void * args,
        const size_t args_size,
        const task_accesses_setter_t & set_accesses,
        const task_access_counter_t naccesses
    ) {
        assert(naccesses > 0);
        assert(set_accesses);

        // retrieve tls
        thread_t * tls = thread_t::get_tls();
        assert(tls);

        // create the task
        const size_t task_size = task_compute_size(flags, naccesses);
        task_t * task = this->task_new(fmtid, flags, task_size + args_size);

        task_dep_info_t * dep = TASK_DEP_INFO(task);
        new (dep) task_dep_info_t(naccesses);

        void * task_args = TASK_ARGS(task, task_size);
        assert(task_args);
        memcpy(task_args, args, args_size);

        access_t * accesses = TASK_ACCESSES(task, flags);
        set_accesses(task, accesses);
        tls->resolve(accesses, naccesses);

        # if XKRT_SUPPORT_DEBUG
        snprintf(task->label, sizeof(task->label), "capture-dynamic-access");
        # endif

        return task;
    }

    template <task_flag_bitfield_t flags, size_t args_size>
    inline task_t *
    task_instanciate(
        const task_format_id_t fmtid,
        const task_accesses_setter_t & set_accesses,
        const task_access_counter_t naccesses
    ) {
        return this->task_instanciate<flags>(fmtid, NULL, args_size, set_accesses, naccesses);
    }

    template <task_access_counter_t ac, bool has_set_accesses, bool has_split_condition>
    inline task_t *
    task_instanciate(
        const task_routine_t & f,
        const task_accesses_setter_t & set_accesses,
        const std::function<bool(task_t *, access_t *)> & split_condition
    ) {
        static_assert(ac == 0 || has_set_accesses);     // must have both or none
        static_assert(!has_split_condition || ac > 0);  // cannot split if task has no accesses

        assert(has_set_accesses    == (set_accesses    != nullptr));
        assert(has_split_condition == (split_condition != nullptr));

        // retrieve tls
        thread_t * tls = thread_t::get_tls();
        assert(tls);

        // create the task
        constexpr task_flag_bitfield_t depflag = ac                  ? TASK_FLAG_DEPENDENT : TASK_FLAG_ZERO;
        constexpr task_flag_bitfield_t molflag = has_split_condition ? TASK_FLAG_MOLDABLE  : TASK_FLAG_ZERO;
        constexpr task_flag_bitfield_t flags = depflag | molflag;
        constexpr size_t task_size = task_compute_size(flags, ac);
        constexpr size_t args_size = sizeof(f);

        task_t * task = this->task_new(this->formats.host_capture, flags, task_size + args_size);

        std::function<void(runtime_t *, device_t *, task_t *)> * fcpy = (std::function<void(runtime_t *, device_t *, task_t *)> *) TASK_ARGS(task, task_size);
        new (fcpy) std::function<void(runtime_t *, device_t *, task_t *)>(f);

        if (depflag)
        {
            task_dep_info_t * dep = TASK_DEP_INFO(task);
            new (dep) task_dep_info_t(ac);

            access_t * accesses = TASK_ACCESSES(task, flags);
            set_accesses(task, accesses);
            tls->resolve(accesses, ac);
        }

        if (molflag)
        {
            task_mol_info_t * mol = TASK_MOL_INFO(task);
            new (mol) task_mol_info_t(split_condition, args_size);
        }

        # if XKRT_SUPPORT_DEBUG
        snprintf(task->label, sizeof(task->label), "capture");
        # endif

        return task;
    }

    /**
     * @brief Spawn a task in the currently executing thread team
     * @tparam ac Task access counter specifying the number of data accesses
     * @tparam has_set_accesses Flag indicating if access setter is provided
     * @tparam has_split_condition Flag indicating if split condition is provided
     * @param set_accesses Function to set task data accesses
     * @param split_condition Function to determine if task should be split (for moldable tasks)
     * @param f Task execution function
     */
    template <task_access_counter_t ac, bool has_set_accesses, bool has_split_condition>
    inline void
    task_spawn(
        const task_accesses_setter_t & set_accesses,
        const std::function<bool(task_t *, access_t *)> & split_condition,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        // create the task
        task_t * task = this->task_instanciate<ac, has_set_accesses, has_split_condition>(f, set_accesses, split_condition);
        assert(task);

        // commit the task
        thread_t * tls = thread_t::get_tls();
        assert(tls);

        tls->commit(task, task_enqueue, this);
    }

    /**
     * @brief Spawn a task with data accesses and split condition
     * @tparam ac Task access counter specifying the number of data accesses
     * @param set_accesses Function to set task data accesses
     * @param split_condition Function to determine if task should be split
     * @param f Task execution function
     */
    template <task_access_counter_t ac>
    inline void
    task_spawn(
        const task_accesses_setter_t & set_accesses,
        const std::function<bool(task_t *, access_t *)> & split_condition,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        return this->task_spawn<ac, true, true>(set_accesses, split_condition, f);
    }

    /**
     * @brief Spawn a task with data accesses but no split condition
     * @tparam ac Task access counter specifying the number of data accesses
     * @param set_accesses Function to set task data accesses
     * @param f Task execution function
     */
    template <task_access_counter_t ac>
    inline void
    task_spawn(
        const task_accesses_setter_t & set_accesses,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        this->task_spawn<ac, true, false>(set_accesses, nullptr, f);
    }

    /**
     * @brief Spawn a simple task with no data accesses
     * @param f Task execution function
     */
    inline void
    task_spawn(
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        this->task_spawn<0, false, false>(nullptr, nullptr, f);
    }

    /////////////////////////
    // THREADING - THREADS //
    /////////////////////////

    /**
     * @brief Retrieve the CPU affinity mask of the calling thread
     * @param[out] cpuset CPU set to be filled with the current thread's affinity
     */
    static void thread_getaffinity(cpu_set_t & cpuset);

    /**
     * @brief Bind the calling thread to the specified CPU set
     * @param cpuset CPU set defining which CPUs the thread can run on
     */
    static void thread_setaffinity(cpu_set_t & cpuset);

    ///////////////////////
    // THREADING - TEAMS //
    ///////////////////////

    /**
     * @brief Create a new thread team
     * @param team Pointer to the team structure to initialize
     */
    void team_create(team_t * team);

    /**
     * @brief Synchronization barrier - wait until all threads reach this point
     * @tparam worksteal Enable work stealing during barrier wait
     * @param team Pointer to the team
     * @param thread Pointer to the calling thread (optional)
     */
    template<bool worksteal = false>
    void team_barrier(team_t * team, thread_t * thread = NULL);

    /**
     * @brief Wait until all threads have finished and destroy the team
     * @param team Pointer to the team to join and destroy
     */
    void team_join(team_t * team);

    /**
     * @brief Enter a critical section (mutual exclusion)
     * @param team Pointer to the team
     */
    void team_critical_begin(team_t * team);

    /**
     * @brief Exit a critical section
     * @param team Pointer to the team
     */
    void team_critical_end(team_t * team);

    /* functor for parallel for */
    typedef std::function<void(thread_t * thread)> team_parallel_for_func_t;
    typedef std::function<void(thread_t * thread, const int i)> team_parallel_for_i_func_t;

    /**
     * @brief Execute a parallel for loop across team threads (blocking)
     * @param team Pointer to the team
     * @param func Function to execute in parallel by each thread
     */
    void team_parallel_for(team_t * team, team_parallel_for_func_t func);

    inline void
    team_parallel_for(
        team_t * team,
        team_parallel_for_i_func_t func,
        const int up,
        const int low = 0,
        const int incr = 1
    ) {
        this->team_parallel_for(team, [=] (thread_t * thread) {
                int up_v = up;
                int low_v = low;
                int last_iter;
                team_t::parallel_for_thread_bounds(&last_iter, &low_v, &up_v, incr);
                if (incr > 0)
                {
                    for (int i = low_v ; i <= up_v ; i += incr)
                        func(thread, i);
                }
                else
                {
                    LOGGER_FATAL("Not supported");
                }

            }
        );
    }

    template <int UP, int LOW = 0, int INCR = 1>
    inline void
    team_parallel_for(team_t * team, team_parallel_for_i_func_t func)
    {
        return team_parallel_for(team, func, UP, LOW, INCR);
    }

    /////////////////////////
    // THREADING - TASKING //
    /////////////////////////

    /**
     * @brief Spawn a task within a specific team with given accesses
     * @param team Target team for task execution
     * @param set_accesses Function to set task data accesses
     * @param f Task execution function
     * @param naccesses Number of data accesses (must be > 0)
     */
    inline void
    team_task_spawn(
        team_t * team,
        const task_routine_t & f,
        const task_accesses_setter_t & set_accesses,
        const task_access_counter_t naccesses
    ) {
        assert(naccesses > 0);

        // create the task
        constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT;
        constexpr size_t args_size = sizeof(task_routine_t);
        const size_t task_size = task_compute_size(flags, naccesses);
        task_t * task = task_instanciate<flags, args_size>(this->formats.host_capture, set_accesses, naccesses);
        assert(task);

        task_routine_t * routine = (task_routine_t *) TASK_ARGS(task, task_size);
        new (routine) task_routine_t(f);

        // commit the task
        thread_t * tls = thread_t::get_tls();
        tls->commit(task, task_team_enqueue, this, team);
    }

    /**
     * @brief Spawn a task within a specific team with given accesses
     * @param team Target team for task execution
     * @param set_accesses Function to set task data accesses
     * @param f Task execution function
     * @param naccesses Number of data accesses (must be > 0)
     */
    inline void
    team_task_spawn(
        team_t * team,
        const task_format_id_t fmtid,
        const void * args,
        const size_t args_size,
        const task_accesses_setter_t & set_accesses,
        const task_access_counter_t naccesses
    ) {
        assert(naccesses > 0);

        // create the task
        constexpr task_flag_bitfield_t flags = TASK_FLAG_DEPENDENT | TASK_FLAG_DETACHABLE;
        task_t * task = task_instanciate<flags>(fmtid, args, args_size, set_accesses, naccesses);
        assert(task);

        // commit the task
        thread_t * tls = thread_t::get_tls();
        tls->commit(task, task_team_enqueue, this, team);
    }

    /**
     * @brief Spawn a task within a specific team
     * @tparam ac Task access counter specifying the number of data accesses
     * @tparam has_set_accesses Flag indicating if access setter is provided
     * @tparam has_split_condition Flag indicating if split condition is provided
     * @param team Target team for task execution
     * @param set_accesses Function to set task data accesses
     * @param split_condition Function to determine if task should be split
     * @param f Task execution function
     */
    template <task_access_counter_t ac, bool has_set_accesses, bool has_split_condition>
    inline void
    team_task_spawn(
        team_t * team,
        const task_accesses_setter_t & set_accesses,
        const std::function<bool(task_t *, access_t *)> & split_condition,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        // create the task
        task_t * task = task_instanciate<ac, has_set_accesses, has_split_condition>(f, set_accesses, split_condition);
        assert(task);

        // commit the task
        thread_t * tls = thread_t::get_tls();
        tls->commit(task, task_team_enqueue, this, team);
    }

    /**
     * @brief Spawn a team task with data accesses and split condition
     * @tparam ac Task access counter specifying the number of data accesses
     * @param team Target team for task execution
     * @param set_accesses Function to set task data accesses
     * @param split_condition Function to determine if task should be split
     * @param f Task execution function
     */
    template <task_access_counter_t ac>
    inline void
    team_task_spawn(
        team_t * team,
        const task_accesses_setter_t & set_accesses,
        const std::function<bool(task_t *, access_t *)> & split_condition,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        return this->team_task_spawn<ac, true, true>(team, set_accesses, split_condition, f);
    }

    /**
     * @brief Spawn a team task with data accesses but no split condition
     * @tparam ac Task access counter specifying the number of data accesses
     * @param team Target team for task execution
     * @param set_accesses Function to set task data accesses
     * @param f Task execution function
     */
    template <task_access_counter_t ac>
    inline void
    team_task_spawn(
        team_t * team,
        const task_accesses_setter_t & set_accesses,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        this->team_task_spawn<ac, true, false>(team, set_accesses, nullptr, f);
    }

    /**
     * @brief Spawn a simple team task with no data accesses
     * @param team Target team for task execution
     * @param f Task execution function
     */
    inline void
    team_task_spawn(
        team_t * team,
        const std::function<void(runtime_t *, device_t *, task_t *)> & f
    ) {
        this->team_task_spawn<0, false, false>(team, nullptr, nullptr, f);
    }

    //////////////////
    // TEAM - UTILS //
    //////////////////

    /**
     * @brief Retrieve the thread team for a specific device
     * @param type Driver type (e.g., CPU, GPU)
     * @param device_driver_id Device identifier within the driver
     * @return Pointer to the team, or nullptr if not found
     */
    team_t * team_get(const driver_type_t type, const device_driver_id_t device_driver_id);

    /**
     * @brief Retrieve the thread team for a specific driver type
     * @param type Driver type (e.g., CPU, GPU)
     * @return Pointer to the team, or nullptr if not found
     */
    team_t * team_get(const driver_type_t type);

    /**
     * @brief Retrieve the first available team from multiple driver types
     * @param types Bitfield of driver types to search
     * @return Pointer to the first non-null team found, or nullptr
     */
    team_t * team_get_any(const driver_type_bitfield_t types);

    ////////////
    // ENERGY //
    ////////////

    /**
     * @brief Start recording energy consumption for a device
     * @param device_global_id Global identifier of the device
     * @param[out] pwr Power measurement structure to initialize
     */
    void power_start(const device_global_id_t device_global_id, power_t * pwr);

    /**
     * @brief Stop recording and retrieve energy consumption
     * @param device_global_id Global identifier of the device
     * @param[in,out] pwr Power measurement structure to finalize
     */
    void power_stop(const device_global_id_t device_global_id, power_t * pwr);

    ///////////////
    // UTILITIES //
    ///////////////

    /**
     * @brief Get a driver by type
     * @param type Driver type to retrieve
     * @return Pointer to the driver, or nullptr if not found
     */
    driver_t * driver_get(const driver_type_t type);

    /**
     * @brief Get a device by its global identifier
     * @param device_global_id Global device identifier
     * @return Pointer to the device, or nullptr if not found
     */
    device_t * device_get(const device_global_id_t device_global_id);

    /**
     * @brief Get bitfield of all devices for a given driver type
     * @param type Driver type
     * @return Bitfield with bits set for each available device
     */
    device_global_id_bitfield_t devices_get(const driver_type_t type);

    /**
     * @brief Get the number of committed (active) devices
     * @return Number of devices currently in use
     */
    unsigned int get_ndevices(void);

    /**
     * @brief Get the maximum number of devices available
     * @return Maximum device count
     */
    unsigned int get_ndevices_max(void);

    //////////////////
    // TASK FORMATS //
    //////////////////

    /**
     * @brief Initializes the task formats repository.
     *
     * Sets up the ::task_formats_t structure, likely by zeroing
     * memory and setting `next_fmtid` to a starting value (e.g., 1,
     * as ::XKRT_TASK_FORMAT_NULL is 0).
     *
     * @param formats Pointer to the ::task_formats_t instance to initialize.
     */
    void task_formats_init(void);

    /**
     * @brief Creates and registers a new task format.
     *
     * Copies the provided `format` data into the `formats` list at the
     * next available ID.
     *
     * @param formats Pointer to the ::task_formats_t repository.
     * @param format  Pointer to the ::task_format_t to register.
     * @return The new ::task_format_id_t assigned to this format.
     */
    task_format_id_t task_format_create(const task_format_t * format);

    /**
     * @brief Allocates a new task format ID without setting its data.
     *
     * This function likely reserves a new ID by incrementing `next_fmtid`
     * and returning its previous value. The caller is then responsible for
     * setting the format's data, perhaps using ::task_format_set.
     *
     * @param formats Pointer to the ::task_formats_t repository.
     * @param label   The label of the task format
     * @return The allocated ::task_format_id_t.
     */
    task_format_id_t task_format_put(const char * label);

    /**
     * @brief Sets the function for a specific target on the passed task format.
     *
     * @param formats Pointer to the ::task_formats_t repository.
     * @param fmtid   The ::task_format_id_t of the format to update.
     * @param target  The ::task_format_target_t to set the function for.
     * @param func    The ::task_format_func_t implementation for that target.
     * @return 0 on success, or an error code if failed.
     */
    int task_format_set(task_format_id_t fmtid, task_format_target_t target, task_format_func_t func);

    /**
     * @brief Retrieves a task format by its ID.
     *
     * @param formats Pointer to the ::task_formats_t repository.
     * @param fmtid      The ::task_format_id_t of the format to retrieve.
     * @return A pointer to the corresponding ::task_format_t, or NULL
     * if the ID is invalid (e.g., ::XKRT_TASK_FORMAT_NULL or out of bounds).
     */
    task_format_t * task_format_get(task_format_id_t fmtid);

    /**
     *  Single kernel launcher.
     *
     *  Submit a kernel launch command to the device, then return.
     *  The 'launch' is executed by a thread of the device implicit team.
     *  If 'synchronous' is set to true, the runtime the kernel to be executed after returning from the launcher (i.e., synchronous execution).
     *  Else, it expect the launcher to associate an event to the kernel completion. (i.e., executed asynchronously)
     *  The task must be detachable, and completion is associated with the command completion.
     *
     *  'device' is the targeted device
     *  'task' is the currently executing task
     *  'attach_event' if the launcher will attach an event to the command
     *  'launcher' is the kernel launcher
     */
    template <bool synchronous>
    void task_kernel_launch(
        device_t * device,
        task_t * task,
        kernel_launcher_t launcher
    );

    # if XKRT_SUPPORT_STATS

    ///////////
    // STATS //
    ///////////

    /**
     * @brief Runtime statistics collection
     */
    struct {
        struct {
            stats_int_t submitted;   ///< Number of tasks submitted
            stats_int_t commited;    ///< Number of tasks committed
            stats_int_t completed;   ///< Number of tasks completed
        } tasks[XKRT_TASK_FORMAT_MAX];

        struct {
            stats_int_t registered;     ///< Memory regions registered
            stats_int_t unregistered;   ///< Memory regions unregistered
            struct {
                struct {
                    stats_int_t device;  ///< Device-side advised allocations
                    stats_int_t host;    ///< Host-side advised allocations
                } advised;
                struct {
                    stats_int_t device;  ///< Device-side prefetched data
                    stats_int_t host;    ///< Host-side prefetched data
                } prefetched;
            } unified;
        } memory;
    } stats;

    /**
     * @brief Report collected runtime statistics
     */
    void stats_report(void);

    # endif /* XKRT_SUPPORT_STATS */

}; /* runtime_t */

/**
 * @brief Main routine for parallel for execution
 * @param runtime Pointer to the runtime system
 * @param team Pointer to the team
 * @param thread Pointer to the thread
 * @return Thread return value
 */
void * team_parallel_for_main(runtime_t * runtime, team_t * team, thread_t * thread);

/// Macro defining the parallel for team routine function pointer
# define XKRT_TEAM_ROUTINE_PARALLEL_FOR ((team_routine_t) team_parallel_for_main)

XKRT_NAMESPACE_END

#endif /* __XKRT_RUNTIME_H__ */
