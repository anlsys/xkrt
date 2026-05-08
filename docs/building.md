# Building & Configuration {#building}

> **Note**: This documentation was generated with the assistance of AI (Claude Opus 4.6 via OpenCode) and verified by the authors.

---

## Requirements

- **C/C++ compiler** with C++20 support. The only tested compiler is **LLVM/Clang >= 20.x**.
- **CMake** >= 3.17
- **hwloc** -- hardware locality library ([github.com/open-mpi/hwloc](https://github.com/open-mpi/hwloc))

## Optional Dependencies

### GPU Backends

| Backend | CMake Option | Library |
|---------|-------------|---------|
| CUDA | `-DUSE_CUDA=on` | NVIDIA CUDA Driver API |
| HIP | `-DUSE_HIP=on` | AMD ROCm HIP |
| Level Zero | `-DUSE_ZE=on` | Intel oneAPI Level Zero |
| SYCL | `-DUSE_SYCL=on` | SYCL (e.g., Intel DPC++) |
| OpenCL | `-DUSE_CL=on` | OpenCL |

### BLAS Libraries

| Library | Description |
|---------|-------------|
| CUBLAS | NVIDIA cuBLAS (auto-detected with CUDA) |
| HIPBLAS | AMD hipBLAS (auto-detected with HIP) |
| ONEAPI::MKL | Intel oneMKL (with SYCL) |

### Management / Monitoring

| Library | Description |
|---------|-------------|
| NVML | NVIDIA Management Library (GPU power/temperature) |
| RSMI | AMD ROCm SMI (GPU power/temperature) |
| Level Zero Sysman | Intel Level Zero system management |

### Other

| Library | Description |
|---------|-------------|
| AML | Argonne Memory Library ([github.com/anlsys/aml](https://github.com/anlsys/aml)) |
| Cairo | For debug visualization of memory trees |
| Protheus | JIT compilation support |
| Julia | Julia language bindings |

---

## Build Examples

### Host-Only (Development / No GPU)

```bash
mkdir build && cd build
CC=clang CXX=clang++ cmake \
    -DUSE_STATS=on \
    -DCMAKE_BUILD_TYPE=Debug \
    ..
make -j$(nproc)
```

### With CUDA Support

```bash
mkdir build && cd build
CC=clang CXX=clang++ \
    CMAKE_PREFIX_PATH=$CUDA_PATH:$CMAKE_PREFIX_PATH \
    cmake -DUSE_CUDA=on ..
make -j$(nproc)
```

### With CUDA (Optimized Release)

```bash
mkdir build && cd build
CC=clang CXX=clang++ \
    CMAKE_PREFIX_PATH=$CUDA_PATH:$CMAKE_PREFIX_PATH \
    cmake \
    -DUSE_CUDA=on \
    -DUSE_SHUT_UP=on \
    -DENABLE_HEAVY_DEBUG=off \
    -DCMAKE_BUILD_TYPE=Release \
    ..
make -j$(nproc)
```

### With Multiple Backends

```bash
CC=clang CXX=clang++ cmake \
    -DUSE_CUDA=on \
    -DUSE_HIP=on \
    -DUSE_ZE=on \
    ..
```

---

## CMake Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `USE_CUDA` | `off` | Enable NVIDIA CUDA backend |
| `USE_HIP` | `off` | Enable AMD HIP backend |
| `USE_ZE` | `off` | Enable Intel Level Zero backend |
| `USE_SYCL` | `off` | Enable SYCL backend |
| `USE_CL` | `off` | Enable OpenCL backend |
| `USE_STATS` | `off` | Enable runtime statistics collection |
| `USE_SHUT_UP` | `off` | Suppress informational log messages |
| `USE_JULIA` | `off` | Enable Julia bindings |
| `USE_CAIRO` | `off` | Enable Cairo debug visualization |
| `USE_PROTHEUS` | `off` | Enable JIT compilation via Protheus |
| `ENABLE_HEAVY_DEBUG` | `on` (Debug) | Enable expensive debug assertions |
| `STRICT` | `off` | Treat warnings as errors |
| `BUILD_C_API` | `on` | Build the C API wrapper (`libxkrt_c.so`) |
| `CMAKE_BUILD_TYPE` | -- | `Debug`, `Release`, `RelWithDebInfo` |

---

## Build Outputs

XKRT builds as a set of shared libraries:

| Library | Description |
|---------|-------------|
| `libxkrt.so` | Core runtime library |
| `libxkrt_driver_host.so` | Host CPU driver (always built) |
| `libxkrt_driver_cu.so` | CUDA driver (if `USE_CUDA=on`) |
| `libxkrt_driver_hip.so` | HIP driver (if `USE_HIP=on`) |
| `libxkrt_driver_ze.so` | Level Zero driver (if `USE_ZE=on`) |
| `libxkrt_driver_sycl.so` | SYCL driver (if `USE_SYCL=on`) |
| `libxkrt_driver_cl.so` | OpenCL driver (if `USE_CL=on`) |

---

## Installation

```bash
cmake --install build --prefix /your/install/path
```

This installs:
- Headers to `<prefix>/include/xkrt/`
- Libraries to `<prefix>/lib/`
- CMake config files to `<prefix>/lib/cmake/xkrt/`

### Using XKRT from CMake

After installation, use `find_package`:

```cmake
find_package(xkrt REQUIRED)
target_link_libraries(my_app PRIVATE xkrt::xkrt)
```

---

## Running Tests

```bash
cd build
ctest --output-on-failure
```

The test suite covers:
- Runtime init/deinit
- Task synchronization and dependencies (handle, interval, matrix, graph)
- Task formats
- Memory registration (sync, async, parallel, touch)
- Thread teams (barrier, parallel for, device teams)
- Moldable tasks
- File I/O
- C API

---

## Environment Variables

Set `XKRT_HELP=1` when running any XKRT application to display all available environment variables and their current values:

```bash
XKRT_HELP=1 ./my_app
```
