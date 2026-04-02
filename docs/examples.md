# Examples {#examples}

> **Note**: This documentation was generated with the assistance of AI (Claude Opus 4.6 via OpenCode) and verified by the authors.

This page walks through the example programs shipped in the `examples/` directory.

---

## AXPY -- Vector Addition with Data Accesses

**File**: `examples/axpy.cc`

This example computes `z[i] = x[i] + y[i]` for 32,768 elements, spawning one task per element. Each task declares three segment accesses (two reads and one write), and the runtime resolves dependencies automatically.

```cpp
#include <xkrt/runtime.h>
XKRT_NAMESPACE_USE;

#define BYTE unsigned char

static void func(task_t * task) {
    access_t * accesses = TASK_ACCESSES(task);
    BYTE * x = (BYTE *) (accesses + 0)->host_view.addr;
    BYTE * y = (BYTE *) (accesses + 1)->host_view.addr;
    BYTE * z = (BYTE *) (accesses + 2)->host_view.addr;
    *z = *x + *y;
}

int main(int argc, char ** argv) {
    runtime_t runtime;
    runtime.init();

    int N = 32768;
    uintptr_t x = (uintptr_t) calloc(1, N);
    uintptr_t y = (uintptr_t) calloc(1, N);
    uintptr_t z = (uintptr_t) calloc(1, N);

    // Initialize x, y with random values...

    for (int i = 0; i < N; ++i) {
        runtime.task_spawn<3>(
            // Declare 3 accesses: read x[i], read y[i], write z[i]
            [&x, &y, &z, &i] (task_t * task, access_t * accesses) {
                new (accesses + 0) access_t(task, x+i, x+(i+1), ACCESS_MODE_R);
                new (accesses + 1) access_t(task, y+i, y+(i+1), ACCESS_MODE_R);
                new (accesses + 2) access_t(task, z+i, z+(i+1), ACCESS_MODE_W);
            },
            // Task body
            [] (runtime_t * runtime, device_t * device, task_t * task) {
                func(task);
            }
        );
    }

    // Wait for all tasks to complete
    runtime.task_wait();

    // Verify results...
    runtime.deinit();
    return 0;
}
```

### Key concepts demonstrated

- **`task_spawn<N>`**: the template parameter `N` is the number of data accesses the task declares.
- **Placement-new for accesses**: accesses are constructed in-place using `new (accesses + i) access_t(...)`.
- **Segment accesses**: each access describes a contiguous byte range `[begin, end)` and a mode (`ACCESS_MODE_R`, `ACCESS_MODE_W`).
- **`task_wait()`**: blocks until all child tasks of the current task complete.

---

## AXPBY -- Moldable Tasks with Split Conditions

**File**: `examples/task-axpby.cc`

Same computation as AXPY, but demonstrates **moldable tasks**. Instead of spawning one task per element, a single task covering the entire vector is spawned with a split condition. The runtime recursively splits it into sub-tasks until the condition is no longer met.

```cpp
runtime.task_spawn<3>(
    // Accesses covering the full vectors
    [&x, &y, &z, &N] (task_t * task, access_t * accesses) {
        new (accesses + 0) access_t(task, x, x + N, ACCESS_MODE_R);
        new (accesses + 1) access_t(task, y, y + N, ACCESS_MODE_R);
        new (accesses + 2) access_t(task, z, z + N, ACCESS_MODE_W);
    },

    // Split condition: keep splitting while the segment has more than 1 element
    [] (task_t * task, access_t * accesses) {
        return (accesses + 0)->host_view.m > 1;
    },

    // Task body (executes on leaf tasks)
    [] (runtime_t * runtime, device_t * device, task_t * task) {
        func(task);
    }
);
```

### Key concepts demonstrated

- **Moldable tasks**: the runtime automatically splits a task based on a user-defined predicate, creating a tree of sub-tasks.
- **Split condition**: a function `(task_t*, access_t*) -> bool` that returns `true` if the task should be subdivided further.
- **Automatic access subdivision**: when a moldable task is split, the runtime bisects each access's memory region.

---

## GPU Kernel Launch

**File**: `examples/task-device.cc`

This example shows how to target a GPU device and launch a kernel from within a task routine.

```cpp
#include <xkrt/runtime.h>
XKRT_NAMESPACE_USE;

int main(int argc, char ** argv) {
    runtime_t runtime;
    runtime.init();

    using TYPE = double;
    constexpr int N = 64;
    TYPE x[N];
    memset(x, 0, sizeof(x));

    // Target GPU device 1
    constexpr device_global_id_t gpu_device_global_id = 1;

    runtime.task_spawn<1>(
        gpu_device_global_id,

        // Declare a read access on array x
        [&x] (task_t * task, access_t * accesses) {
            new (accesses + 0) access_t(task, x, N, sizeof(TYPE), ACCESS_MODE_R);
        },

        // Task routine: runs on a thread associated with the GPU
        [] (runtime_t * runtime, device_t * device, task_t * task) {
            access_t * accesses = TASK_ACCESSES(task);
            uintptr_t x_dev = (accesses + 0)->device_view.addr;

            // Launch a kernel asynchronously
            runtime->task_detachable_kernel_launch(
                device, task,
                [] (runtime_t * runtime, device_t * device, task_t * task,
                    queue_t * queue, command_t * command,
                    queue_command_list_counter_t event) {
                    driver_t * driver = runtime->driver_get(device->driver_type);
                    const driver_module_fn_t * fn = nullptr; // your kernel function
                    driver->f_kernel_launch(queue, event, fn,
                        1,1,1, 1,1,1, 0, NULL, 0);
                }
            );
        }
    );

    runtime.deinit();
    return 0;
}
```

### Key concepts demonstrated

- **Device targeting**: passing a `device_global_id_t` as the first argument to `task_spawn` targets a specific device. Device `0` is always the host; device `1+` are GPUs.
- **`device_view.addr`**: inside the task routine, `accesses->device_view.addr` gives the device-side address of the replica. The runtime has already ensured coherence (data transferred from host to device).
- **Asynchronous kernel launch**: `task_detachable_kernel_launch` submits a kernel to a device queue. The task does not complete until the kernel finishes (tracked via the detach counter).
- **Driver abstraction**: `driver->f_kernel_launch(...)` is the driver-agnostic kernel launch interface, supporting CUDA, HIP, Level Zero, SYCL, and OpenCL.

---

## Fibonacci -- Recursive Tasking with Teams

**File**: `examples/fib/main.cc`

Classic recursive Fibonacci using task parallelism. Demonstrates creating a thread team and spawning tasks without data accesses.

```cpp
#include <xkrt/runtime.h>
XKRT_NAMESPACE_USE;

#define CUTOFF_DEPTH 10
static runtime_t runtime;

static inline int fib(int n, int depth = 0) {
    if (n <= 2)
        return n;

    int fn1, fn2;
    if (depth >= CUTOFF_DEPTH) {
        // Sequential below cutoff
        fn1 = fib(n-1, depth+1);
        fn2 = fib(n-2, depth+1);
    } else {
        // Parallel above cutoff
        runtime.task_spawn(
            [&n, &fn1, depth] (runtime_t * rt, device_t * dev, task_t * task) {
                fn1 = fib(n - 1, depth + 1);
            }
        );
        runtime.task_spawn(
            [&n, &fn2, depth] (runtime_t * rt, device_t * dev, task_t * task) {
                fn2 = fib(n - 2, depth + 1);
            }
        );
        runtime.task_wait();
    }
    return fn1 + fn2;
}

static void * main_team(runtime_t * rt, team_t * team, thread_t * thread) {
    if (thread->tid == 0) {
        int r = fib(N);
        runtime.task_wait();
        assert(r == fib_values[N]);
    }
    runtime.team_barrier<true>(team, thread);
    return NULL;
}

int main(int argc, char ** argv) {
    runtime.init();

    team_t team;
    team.desc.routine = (team_routine_t) main_team;
    runtime.team_create(&team);
    runtime.team_join(&team);

    runtime.deinit();
    return 0;
}
```

### Key concepts demonstrated

- **`task_spawn(routine)`**: the simplest spawn overload -- no accesses, no device targeting.
- **`task_wait()`**: used for fork-join parallelism; waits for the two child tasks before summing results.
- **Thread teams**: `team_create` / `team_join` manage a pool of worker threads that cooperatively execute tasks. The team routine runs on all threads; typically thread 0 drives the computation while others participate via work stealing.
- **`team_barrier<true>`**: barrier with work-stealing enabled, so idle threads help process pending tasks.
- **Cutoff depth**: standard optimization to avoid excessive task creation overhead for small subproblems.

---

## File-to-GPU I/O Pipeline

**File**: `examples/io/xkrt.cc`

Benchmarks pipelined file I/O to GPU memory using XKRT's parallel async primitives.

```cpp
#include <xkrt/runtime.h>
XKRT_NAMESPACE_USE;

int main(int argc, char ** argv) {
    const char * filename = argv[1];
    const int NTASKS = atoi(argv[2]);

    int fd = open(filename, O_RDONLY | O_DIRECT);
    off_t size = get_file_size(fd);

    void * buffer = mmap(nullptr, size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    runtime_t runtime;
    runtime.init();
    const device_global_id_t device_global_id = 1;

    // Pipeline: register -> read -> transfer to GPU
    runtime.memory_register_async(buffer, size, NTASKS);
    runtime.file_read_async(fd, buffer, size, NTASKS);
    runtime.memory_coherent_async(device_global_id, buffer, size, NTASKS);
    runtime.task_wait();

    runtime.memory_unregister(buffer, size);
    runtime.deinit();
    return 0;
}
```

### Key concepts demonstrated

- **`memory_register_async(ptr, size, n)`**: pin host memory in parallel using `n` tasks. Pinning is required for DMA transfers to GPUs.
- **`file_read_async(fd, buffer, size, n)`**: read a file in parallel using `n` tasks. Each task reads a chunk at the correct file offset.
- **`memory_coherent_async(device_id, ptr, size, n)`**: transfer data to the GPU in parallel. The runtime's coherence protocol handles the actual H2D copies.
- **Pipelining**: because these operations declare data accesses on the same memory segments, the runtime overlaps pinning, reading, and transferring at the chunk granularity -- a chunk can be transferred to the GPU as soon as it has been read, without waiting for the entire file.

---

## 2D Heat Diffusion Stencil

**Directory**: `examples/heat-diffusion-2d/`

A standalone CMake project that simulates 2D heat diffusion using a stencil pattern. It requires XKRT plus at least one GPU backend (CUDA, HIP, Level Zero, or SYCL).

The simulation:
1. Tiles a 2D grid across available GPUs
2. At each time step, spawns tasks with halo (overlap) accesses for boundary exchange
3. The runtime automatically handles data coherence between GPUs

Kernel implementations are provided for CUDA, HIP, and OpenCL in `examples/heat-diffusion-2d/src/kernels/`.

See `examples/heat-diffusion-2d/README.md` for build instructions.
