# Architecture & Concepts {#architecture}

> **Note**: This documentation was generated with the assistance of AI (Claude Opus 4.6 via OpenCode) and verified by the authors.

This page describes the design principles and key abstractions of the XKRT runtime system.

---

## Overview

XKRT is a **macro-dataflow runtime** that automates memory management alongside parallel task execution on heterogeneous multi-device architectures (CPUs + GPUs). The programmer expresses work as **tasks** with **data accesses**, and the runtime:

1. **Resolves dependencies** between tasks based on overlapping data accesses.
2. **Manages memory coherence** across devices -- allocating replicas, tracking validity, and issuing transfers.
3. **Schedules tasks** on the appropriate device once dependencies are satisfied and data is coherent.

[![XKRT software stack](software-stack.png)](software-stack.pdf)

---

## Software Stack

XKRT sits at the center of the software stack:

- **Above XKRT**: domain-specific libraries like XKBlas (multi-GPU BLAS), XKOMP (OpenMP runtime), and user applications.
- **Below XKRT**: device drivers for CUDA, HIP, Level Zero, SYCL, OpenCL, and the host CPU. XKRT also uses hwloc for hardware topology discovery and optionally NVML/RSMI/Level Zero Sysman for energy monitoring.

---

## Tasks

A **task** is a unit of work with:
- A **routine** -- the function to execute.
- Zero or more **data accesses** -- declared memory regions with a mode (read, write, etc.).
- A set of **flags** controlling its behavior.

Tasks are allocated on a per-thread stack for efficiency (no `malloc` per task). The variable-size layout is computed at compile time from the flag combination.

### Task Flags

Flags are set at task creation time and control which optional info structs are embedded in the task's memory layout. They are combined as a bitfield of type `task_flag_bitfield_t`.

**Layout flags** (set once before instantiation, affect the `task_t` memory layout):

| Flag | Value | Description |
|------|-------|-------------|
| `TASK_FLAG_ZERO` | `0` | No special flags. |
| `TASK_FLAG_ACCESSES` | `1 << 0` | Task declares data accesses. Embeds `task_acs_info_t` and the `access_t` array. |
| `TASK_FLAG_DETACHABLE` | `1 << 1` | Task completion is tied to external event completion (e.g., async kernel launches). Embeds `task_det_info_t` with a reference counter. |
| `TASK_FLAG_DEVICE` | `1 << 2` | Task may execute on a non-host device (GPU). Embeds `task_dev_info_t` with a target `device_unique_id`. Cannot be combined with `TASK_FLAG_DOMAIN`. |
| `TASK_FLAG_DOMAIN` | `1 << 3` | Children tasks of this task may have data dependencies between them. Embeds `task_dom_info_t` with a dependency domain. Cannot be combined with `TASK_FLAG_DEVICE`. |
| `TASK_FLAG_MOLDABLE` | `1 << 4` | Task may be recursively split by the runtime. Embeds `task_mol_info_t` with the split condition and argument size. |
| `TASK_FLAG_GRAPH` | `1 << 5` | Children tasks are recorded into a task dependency graph. Embeds `task_gph_info_t`. |
| `TASK_FLAG_RECORD` | `1 << 6` | Task has a record: emitted commands are buffered for later replay. Embeds `task_rec_info_t`. |

**Runtime flags** (can be set/unset dynamically after creation):

| Flag | Value | Description |
|------|-------|-------------|
| `TASK_FLAG_GRAPH_RECORDING` | `1 << 7` | Currently recording a task dependency graph. |
| `TASK_FLAG_GRAPH_EXECUTE_COMMAND` | `1 << 8` | Recorded commands are also executed (not just buffered). |
| `TASK_FLAG_REQUEUE` | `1 << 9` | Task must be re-queued after returning from its routine. |

### Task States

A task progresses through the following states during its lifetime:

| State | Value | Description |
|-------|-------|-------------|
| `TASK_STATE_ALLOCATED` | 0 | Task memory is allocated and constructed. |
| `TASK_STATE_READY` | 1 | All dependencies are satisfied; data can be fetched. |
| `TASK_STATE_DATA_FETCHING` | 2 | Data is being transferred to the execution device. |
| `TASK_STATE_DATA_FETCHED` | 3 | Data is coherent on the device; routine can execute. |
| `TASK_STATE_EXECUTING` | 4 | The task routine is running. |
| `TASK_STATE_COMPLETED` | 5 | Task completed; successor dependencies can be resolved. |
| `TASK_STATE_DEALLOCATED` | 6 | Task memory is freed (virtual state, never actually set). |

### Task Memory Layout

Tasks use a **variable-size memory layout** determined by their flags:

```
task_t | task_acs_info_t | task_det_info_t | task_dev_info_t |
       | task_dom_info_t | task_mol_info_t | task_gph_info_t |
       | task_rec_info_t | access_t[N]     | args
```

Only the info structs corresponding to set flags are present. A compile-time constexpr switch handles all 128 possible flag combinations, ensuring zero overhead for unused features.

### Moldable Tasks

A **moldable task** is a task that the runtime can recursively split. The programmer provides:
- A **split condition**: `(task_t*, access_t*) -> bool` returning `true` if the task should be subdivided.
- The runtime bisects the task's accesses and creates two child tasks, repeating until the condition is false.

This enables coarse-grained task submission with fine-grained automatic decomposition.

### Task Formats

A **task format** associates a label with per-architecture function pointers. This enables writing multi-target tasks:

```cpp
task_format_id_t fmt = runtime.task_format_put("my_kernel");
runtime.task_format_set(fmt, XKRT_TASK_FORMAT_TARGET_HOST, host_impl);
runtime.task_format_set(fmt, XKRT_TASK_FORMAT_TARGET_CUDA, cuda_impl);
runtime.task_format_set(fmt, XKRT_TASK_FORMAT_TARGET_HIP,  hip_impl);
```

When the runtime schedules a task with this format on a CUDA device, it dispatches to `cuda_impl`; on the host, to `host_impl`.

Available format targets:

| Target | Description |
|--------|-------------|
| `XKRT_TASK_FORMAT_TARGET_HOST` | Host CPU |
| `XKRT_TASK_FORMAT_TARGET_CUDA` | NVIDIA CUDA |
| `XKRT_TASK_FORMAT_TARGET_HIP` | AMD HIP |
| `XKRT_TASK_FORMAT_TARGET_ZE` | Intel Level Zero |
| `XKRT_TASK_FORMAT_TARGET_CL` | OpenCL |
| `XKRT_TASK_FORMAT_TARGET_SYCL` | SYCL |

### Task Dependency Graphs (TDG)

XKRT supports **recording** a sequence of task submissions into a graph, then **replaying** that graph multiple times. This is useful for iterative computations where the same task structure repeats:

```cpp
task_dependency_graph_t tdg;
runtime.task_dependency_graph_record_start(&tdg, true);
// ... spawn tasks ...
runtime.task_wait();
runtime.task_dependency_graph_record_stop();

// Replay the same graph
runtime.task_dependency_graph_replay(&tdg);
runtime.task_dependency_graph_destroy(&tdg);
```

A TDG can also be lowered to a **command graph** for even lower-overhead replay:

```cpp
command_graph_t cg;
runtime.command_graph_from_task_dependency_graph(&tdg, &cg);
runtime.command_graph_replay(&cg);
runtime.command_graph_destroy(&cg);
```

---

## Data Accesses

An **access** declares that a task will touch a memory region with a given mode. Accesses are the foundation of XKRT's automatic dependency resolution and memory coherence.

### Access Modes

| Mode | Value | Semantics |
|------|-------|-----------|
| `ACCESS_MODE_R` | `0b00000001` | Read-only. Multiple concurrent readers are allowed. |
| `ACCESS_MODE_W` | `0b00000010` | Write. Sequential by default (only one writer at a time). |
| `ACCESS_MODE_RW` | `ACCESS_MODE_R \| ACCESS_MODE_W` | Read-write. Combination of read and write. |
| `ACCESS_MODE_V` | `0b00000100` | Virtual/incoherent. Declares a dependency edge without triggering data movement. |
| `ACCESS_MODE_VW` | `ACCESS_MODE_W \| ACCESS_MODE_V` | Virtual write. Write dependency without coherence. |
| `ACCESS_MODE_VR` | `ACCESS_MODE_R \| ACCESS_MODE_V` | Virtual read. Read dependency without coherence. |
| `ACCESS_MODE_D` | `0b00001000` | Detached. Dependencies are not fulfilled on task completion; the task itself is responsible for fulfilling them (used for async kernel completions). |

### Mapping to OpenMP 5.x/6.0 Dependency Types

| OpenMP modifier | XKRT mode | XKRT concurrency |
|-----------------|-----------|-------------------|
| `in` | `ACCESS_MODE_R` | -- |
| `out` / `inout` | `ACCESS_MODE_W` | `ACCESS_CONCURRENCY_SEQUENTIAL` |
| `mutexinoutset` | `ACCESS_MODE_W` | `ACCESS_CONCURRENCY_COMMUTATIVE` |
| `inoutset` | `ACCESS_MODE_W` | `ACCESS_CONCURRENCY_CONCURRENT` |

### Access Concurrency

| Concurrency | Semantics |
|-------------|-----------|
| `ACCESS_CONCURRENCY_SEQUENTIAL` | Default. Strict ordering between writers. |
| `ACCESS_CONCURRENCY_COMMUTATIVE` | Multiple writers may execute in any order, but not concurrently (mutual exclusion). |
| `ACCESS_CONCURRENCY_CONCURRENT` | Multiple writers may execute concurrently (no mutual exclusion). |

### Access Scope

| Scope | Semantics |
|-------|-----------|
| `ACCESS_SCOPE_NONUNIFIED` | Access is resolved in a local dependency domain. |
| `ACCESS_SCOPE_UNIFIED` | Access is resolved in the unified (global) dependency domain. |

### Access Types

| Type | Description |
|------|-------------|
| `ACCESS_TYPE_SEGMENT` | Contiguous 1D memory interval `[begin, end)`. |
| `ACCESS_TYPE_HANDLE` | Opaque handle dependency (address-based, like a pointer). |
| `ACCESS_TYPE_BLAS_MATRIX` | 2D matrix tile described by leading dimension, row/column counts, and element size. |
| `ACCESS_TYPE_NULL` | No memory region (used for pure synchronization dependencies). |

### Dependency Resolution

Dependencies between tasks are computed by the runtime based on overlapping accesses:
- **Read-after-Write (RAW)**: a reader must wait for the previous writer.
- **Write-after-Read (WAR)**: a writer must wait for all previous readers.
- **Write-after-Write (WAW)**: a writer must wait for the previous writer.

The dependency domain uses interval trees (for segment accesses) or hash maps (for handle accesses) to efficiently find overlapping accesses.

---

## Memory Coherence

XKRT maintains a **coherence controller** for each tracked memory region. When a task requires data on a specific device:

1. The coherence controller checks if a valid replica exists on the target device.
2. If not, it finds a valid replica elsewhere (host or another device) and issues a copy command.
3. The replica is marked as valid on the target device, and invalidated on other devices after writes.

This is similar to a cache coherence protocol but operates at the software level across discrete memory spaces.

### Memory Registration (Pinning)

Before DMA transfers can occur between host and device memory, host pages must be **registered** (pinned) with the device drivers. XKRT provides synchronous and asynchronous variants, including parallel registration using `n` tasks for large buffers.

### Data Distribution

XKRT provides utilities to pre-distribute data across devices before computation:

| Distribution type | Description |
|-------------------|-------------|
| `XKRT_DISTRIBUTION_TYPE_CYCLIC1D` | 1D cyclic block distribution. |
| `XKRT_DISTRIBUTION_TYPE_CYCLIC2D` | 2D cyclic distribution. |
| `XKRT_DISTRIBUTION_TYPE_CYCLIC2DBLOCK` | 2D block-cyclic distribution (common for dense linear algebra). |

Distribution is implemented by spawning empty tasks with read accesses, which triggers the coherence protocol to create replicas on the target devices.

---

## Drivers

A **driver** (`driver_t`) abstracts a device backend. Each driver provides function pointers for:
- Device lifecycle (init, finalize, create, destroy)
- Memory management (allocate, deallocate, register, unregister)
- Data transfers (host-to-device, device-to-host, device-to-device, sync and async)
- Kernel launch
- Queue management
- Module loading (for compiled kernels)
- Energy monitoring

Supported driver types:

| Driver type | Description |
|-------------|-------------|
| `XKRT_DRIVER_TYPE_HOST` | Host CPU (always present). |
| `XKRT_DRIVER_TYPE_CUDA` | NVIDIA GPUs via CUDA Driver API. |
| `XKRT_DRIVER_TYPE_HIP` | AMD GPUs via ROCm HIP. |
| `XKRT_DRIVER_TYPE_ZE` | Intel GPUs via Level Zero. |
| `XKRT_DRIVER_TYPE_SYCL` | SYCL-compatible devices. |
| `XKRT_DRIVER_TYPE_CL` | OpenCL devices. |

Drivers are compiled as separate shared libraries (`libxkrt_driver_cu.so`, `libxkrt_driver_hip.so`, etc.) and loaded at runtime.

---

## Thread Teams

A **team** is a group of OS threads that cooperatively execute tasks. Each device has an associated team. Teams support:
- **Work stealing**: idle threads steal tasks from busy threads' deques.
- **Barriers**: synchronize all threads in a team (optionally with work stealing during the wait).
- **Critical sections**: mutual exclusion within a team.
- **Parallel for**: distribute loop iterations across team threads.

---

## Constants

Key compile-time constants defined in `include/xkrt/consts.h`:

| Constant | Value | Description |
|----------|-------|-------------|
| `XKRT_DEVICES_MAX` | 16 | Maximum number of devices in total. |
| `XKRT_DEVICE_MEMORIES_MAX` | 1 | Maximum memory banks per device. |
| `XKRT_THREAD_MAX_MEMORY` | 4 GB | Per-thread task stack size. |
| `XKRT_TASK_MAX_ACCESSES` | 1024 | Maximum data accesses per task. |
| `XKRT_TEAM_MAX_THREADS` | 2048 | Maximum threads per team. |
| `XKRT_HOST_DEVICE_UNIQUE_ID` | 0 | The host device is always device 0. |
| `XKRT_TASK_FORMAT_MAX` | 255 | Maximum number of registered task formats. |

---

## Configuration

XKRT reads configuration from environment variables at startup. Run with `XKRT_HELP=1` to list all available options and their current values.
