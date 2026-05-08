# CMake Minimum version
cmake_minimum_required(VERSION 3.17)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

SET(XKRT xkrt)

# Project name
project(${XKRT})

###############
# C++ version #
###############
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Optional: disables compiler-specific extensions

#########
# RPATH #
#########

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

###########
# OPTIONS #
###########

include(BooleanUtils)

xkoption(ENABLE_HEAVY_DEBUG "Heavy assertion tests in the khp-tree structure, setting this variable to 'ON' will considerably slowdown task insertion, but can help debugging the khp-trees" OFF)
xkoption(STRICT       "Enable strict compilation mode" OFF)
xkoption(BUILD_C_API  "Build the C API" ON)

xkoption(USE_MEMORY_REGISTER_OVERFLOW_PROTECTION "Enable the protection of memory copies over regions that are not continuously registered, by splitting into multiple copies, to avoid cuda crashes" ON)
xkoption(USE_MEMORY_REGISTER_PAGE "Align memory registration request to lower and upper pages." ON)
xkoption(USE_MEMORY_REGISTER_ASSISTED "Allow programmer register/unregister respectively already-registered and non-registered memory." ON)

xkoption(USE_SHUT_UP  "Disable all logging activities" OFF)

xkoption(USE_CL       "Whether to build the OpenCL driver." OFF)
xkoption(USE_CUDA     "Whether to build the CUDA driver." OFF)
xkoption(USE_HIP      "Whether to build the HIP driver." OFF)
xkoption(USE_NVML     "Use Nvidia's NVML management library." OFF)
xkoption(USE_STATS    "Whether to compile the built-in stat system." OFF)
xkoption(USE_SYCL     "Whether to build the SYCL driver." OFF)
xkoption(USE_ZE       "Whether to build the Level Zero driver." OFF)
xkoption(USE_ZES      "Use Intel's ZES management library." OFF)
xkoption(USE_ZE_SYCL_INTEROP "Enable the compilation of SYCL interop with Level Zero driver" OFF)
xkoption(USE_JULIA    "Link with Julia to attach threads (needed for running host tasks from Julia)" OFF)
xkoption(USE_CAIRO    "Use Cairo to generate images (memory-tree debugging purposes mostly)." OFF)
xkoption(USE_PROTHEUS "Use Protheus for JIT." OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(DEBUG_BUILD ON)
else()
    set(DEBUG_BUILD OFF)
endif()
convert_bool(DEBUG_BUILD)

if (USE_MEMORY_REGISTER_ASSISTED AND (NOT USE_MEMORY_REGISTER_OVERFLOW_PROTECTION OR NOT USE_MEMORY_REGISTER_PAGE))
    message(FATAL_ERROR "`USE_MEMORY_REGISTER_ASSISTED` requires both `-DUSE_MEMORY_REGISTER_OVERFLOW_PROTECTION=ON` and `-DUSE_MEMORY_REGISTER_ASSISTED=ON`")
endif()

########
# XKRT #
########

SET(SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/src/access.cc
    ${CMAKE_SOURCE_DIR}/src/command/command-graph.cc
    ${CMAKE_SOURCE_DIR}/src/conf/conf.cc
    ${CMAKE_SOURCE_DIR}/src/debug/cairo.cc
    ${CMAKE_SOURCE_DIR}/src/distribution/distribution.cc
    ${CMAKE_SOURCE_DIR}/src/driver.cc
    ${CMAKE_SOURCE_DIR}/src/driver/device.cc
    ${CMAKE_SOURCE_DIR}/src/driver/driver.cc
    ${CMAKE_SOURCE_DIR}/src/driver/queue.cc
    ${CMAKE_SOURCE_DIR}/src/file/readwrite.cc
    ${CMAKE_SOURCE_DIR}/src/logger/logger.cc
    ${CMAKE_SOURCE_DIR}/src/logger/metric.cc
    ${CMAKE_SOURCE_DIR}/src/memory.cc
    ${CMAKE_SOURCE_DIR}/src/memory/access.cc
    ${CMAKE_SOURCE_DIR}/src/memory/access/handle/dependency-map.cc
    ${CMAKE_SOURCE_DIR}/src/memory/allocator/buddy.cc
    ${CMAKE_SOURCE_DIR}/src/memory/allocator/freelist.cc
    ${CMAKE_SOURCE_DIR}/src/memory/coherent.cc
    ${CMAKE_SOURCE_DIR}/src/memory/copy.cc
    ${CMAKE_SOURCE_DIR}/src/memory/distribute.cc
    ${CMAKE_SOURCE_DIR}/src/memory/invalidate.cc
    ${CMAKE_SOURCE_DIR}/src/memory/register.cc
    ${CMAKE_SOURCE_DIR}/src/power.cc
    ${CMAKE_SOURCE_DIR}/src/runtime.cc
    ${CMAKE_SOURCE_DIR}/src/sched.cc
    ${CMAKE_SOURCE_DIR}/src/stats/stats.cc
    ${CMAKE_SOURCE_DIR}/src/task.cc
    ${CMAKE_SOURCE_DIR}/src/task/format.cc
    ${CMAKE_SOURCE_DIR}/src/task/task-dependency-graph.cc
    ${CMAKE_SOURCE_DIR}/src/task/task.cc
    ${CMAKE_SOURCE_DIR}/src/thread/deque.cc
    ${CMAKE_SOURCE_DIR}/src/thread/thread.cc
)

if (BUILD_C_API)
    list(APPEND SOURCE_FILES ${CMAKE_SOURCE_DIR}/api/c/src/xkrt.cc)
    include_directories(${CMAKE_SOURCE_DIR}/api/c/src)
    include_directories(${CMAKE_SOURCE_DIR}/api/c/include)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/api/c/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
endif()

# XKRT Lib
add_library(${XKRT} SHARED ${SOURCE_FILES})
set_target_properties(${XKRT} PROPERTIES BUILD_RPATH "${CMAKE_SOURCE_DIR}/lib")
set_target_properties(${XKRT} PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

# Include directory
include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_BINARY_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/)

# targets to install (lib)
set(XKRT_TARGETS ${XKRT})

###############
# DEPENDENCES #
###############

# OpenCG

# find_package(OpenCG REQUIRED)
# target_link_libraries(${XKRT} PRIVATE OpenCG::opencg)

add_subdirectory(opencg/)
target_link_libraries(${XKRT} PUBLIC opencg)
include_directories(${CMAKE_SOURCE_DIR}/opencg/include/)

# PTHREAD
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(${XKRT} PRIVATE Threads::Threads)

# HWLOC
find_package(hwloc REQUIRED)
target_link_libraries(${XKRT} PRIVATE hwloc::hwloc)

# Host driver
add_library(${XKRT}_driver_host SHARED src/driver/driver_host.cc)
target_link_libraries(${XKRT} PRIVATE ${XKRT}_driver_host)
list(APPEND XKRT_TARGETS ${XKRT}_driver_host)

# CUDA
if (USE_CUDA)

    find_package(CUDAToolkit REQUIRED)

    add_library(${XKRT}_driver_cu SHARED src/driver/driver_cu.cc)
    target_link_libraries(${XKRT}_driver_cu PRIVATE hwloc::hwloc CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusparse CUDA::cusolver)
    target_link_libraries(${XKRT} PRIVATE ${XKRT}_driver_cu)
    list(APPEND XKRT_TARGETS ${XKRT}_driver_cu)

    # # build driver_cu.cc with nvcc, so nvcc generates its header file
    # enable_language(CUDA)
    # set_source_files_properties(src/driver/driver_cu.cc PROPERTIES LANGUAGE CUDA)
    # # set_source_files_properties(src/driver/driver_cu.cc PROPERTIES COMPILE_OPTIONS "-allow-unsupported-compiler;--compiler-bindir=${CMAKE_CXX_COMPILER}")
    # set_target_properties(${XKRT}_driver_cu PROPERTIES CUDA_ARCHITECTURES "70") # ;61;70;72;75;80;86;89;90")

endif()

# HIP
if (USE_HIP)

    add_library(${XKRT}_driver_hip SHARED src/driver/driver_hip.cc)
    target_link_libraries(${XKRT} PRIVATE ${XKRT}_driver_hip)

    find_package(hip     REQUIRED)
    find_package(hipblas REQUIRED)
    target_link_libraries(${XKRT}_driver_hip PRIVATE hwloc::hwloc hip::host roc::hipblas)

    find_package(RSMI    REQUIRED)
    include_directories("${RSMI_INCLUDE_DIRS}")
    target_link_libraries(${XKRT}_driver_hip PRIVATE ${RSMI_LIBRARIES})

    list(APPEND XKRT_TARGETS ${XKRT}_driver_hip)

endif()

# Level Zero
if (USE_ZE)

    find_package(ZE REQUIRED)
    include_directories("${ZE_INCLUDE_DIRS}")

    add_library(${XKRT}_driver_ze SHARED src/driver/driver_ze.cc)
    target_link_libraries(${XKRT}_driver_ze PRIVATE hwloc::hwloc ${ZE_LIBRARIES})
    target_link_libraries(${XKRT} PRIVATE ${XKRT}_driver_ze)
    list(APPEND XKRT_TARGETS ${XKRT}_driver_ze)

endif()

# SYCL / ZE_SYCL_INTEROP
if (USE_SYCL OR USE_ZE_SYCL_INTEROP)
    find_package(SYCL REQUIRED)
    include_directories(${SYCL_INCLUDE_DIRS})
endif()

if (USE_SYCL)
    add_library(${XKRT}_driver_sycl SHARED src/driver/driver_sycl.cc)
    target_link_libraries( ${XKRT}_driver_sycl PRIVATE ${SYCL_LIBRARIES} -fsycl)
    target_compile_options(${XKRT}_driver_sycl PRIVATE -fsycl)
    target_link_libraries(${XKRT} PRIVATE ${XKRT}_driver_sycl)
    list(APPEND XKRT_TARGETS ${XKRT}_driver_sycl)
endif()

if (USE_ZE_SYCL_INTEROP)
    target_link_libraries( ${XKRT}_driver_ze PRIVATE ${SYCL_LIBRARIES} -fsycl)
    target_compile_options(${XKRT}_driver_ze PRIVATE -fsycl)
endif()

# OpenCL
if (USE_CL)
    find_package(CL REQUIRED)
    include_directories("${CL_INCLUDE_DIRS}")

    add_library(${XKRT}_driver_cl SHARED src/driver/driver_cl.cc)
    target_link_libraries(${XKRT} PRIVATE ${XKRT}_driver_cl ${CL_LIBRARIES})
    list(APPEND XKRT_TARGETS ${XKRT}_driver_cl)
endif()

# NVML
if (USE_NVML)
    find_package(NVML REQUIRED)
    include_directories("${NVML_INCLUDE_DIRS}")
    target_link_libraries(${XKRT} PRIVATE ${NVML_LIBRARIES})
endif()

# ZES (Intel energy)
if (USE_ZES)
    # nothing to do, but must USE_ZE to have any effects
endif()

# Cairo
if (USE_CAIRO)
    find_package(CAIRO REQUIRED)
    include_directories("${CAIRO_INCLUDE_DIRS}")
    target_link_libraries(${XKRT} PRIVATE ${CAIRO_LIBRARIES})
endif()

if (STRICT)
    target_compile_options(${XKRT} PRIVATE
        -Wall          # Common warnings
        -Wextra        # Extra warnings
        -Werror        # Treat warnings as errors
        -Wshadow       # Warn if a variable hides another
        -Wconversion   # Implicit type conversions
        -Wsign-conversion
        -Wformat=2     # Strict format checking
        -Wfloat-equal  # Detect floating point equality checks
        -Wundef        # Warn on undefined macros
        #-Wcast-align   # Warn about pointer alignment
        -Wpointer-arith
        -Wstrict-overflow=5
        -fno-common
        -ftrapv
        -Werror=conversion
        -fstack-protector-strong
        -fstrict-aliasing
        -Wstack-protector
        -Wundef
        #-Wpedantic     # Enforce strict ISO compliance
        #-Wfatal-errors
        -Wmissing-declarations
        -Wmissing-prototypes
        -Wold-style-definition
        -Wno-gnu-anonymous-struct
        -Wno-shadow
        -Wno-nested-anon-types
        -Wno-sign-conversion
        -Wno-int-to-void-pointer-cast
        -Wno-unused-function
        -Wno-unused-parameter
    )
    if (${CUDA_DRIVER_FOUND})
        target_compile_options(${XKRT}_driver_cu PRIVATE -Wall -Wextra -Wundef)
    endif()
endif()

# JULIA
if (USE_JULIA)
    find_package(Julia REQUIRED)
    include_directories(${Julia_INCLUDE_DIRS})
    target_link_libraries(${XKRT} PRIVATE ${Julia_LIBRARIES})
endif()

if (USE_PROTHEUS)
    find_package(proteus CONFIG REQUIRED)
    add_proteus(${XKRT})
endif()

################
# INSTALLATION #
################

# Header file generated
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include/xkrt")
configure_file(
    ${CMAKE_SOURCE_DIR}/include/xkrt/support.h.in
    ${CMAKE_BINARY_DIR}/include/xkrt/support.h
)
install(
    FILES "${CMAKE_BINARY_DIR}/include/xkrt/support.h"
    DESTINATION include/xkrt
)

# Installation - lib and include files
install(TARGETS ${XKRT_TARGETS} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Installation - cmake files
target_include_directories(${XKRT} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
message(STATUS "Targets are ${XKRT_TARGETS}")
install(TARGETS ${XKRT_TARGETS}
        EXPORT XKRTTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT XKRTTargets
        FILE XKRTTargets.cmake
        NAMESPACE XKRT::
        DESTINATION lib/cmake/XKRT)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/XKRTConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/XKRTConfig.cmake"
    INSTALL_DESTINATION lib/cmake/XKRT
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/XKRTConfig.cmake"
    DESTINATION lib/cmake/XKRT
)

################
# Sub projects #
################

add_subdirectory(tests/)
enable_testing()
