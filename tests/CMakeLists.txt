# CMake Minimum version
cmake_minimum_required(VERSION 3.10)
include(CTest)

# Project name
project(tests)

# list of tests
set(TEST_SOURCES

    # C++ tests
    deinit.cc
    fib-task-capture.cc
    fib-task-format.cc
    file-read.cc
    init.cc
    memory-allocator.cc
    memory-register-assisted-async-depend.cc
    memory-register-assisted-async.cc
    memory-register-assisted-unregister.cc
    memory-register-assisted.cc
    memory-register-async-legacy-memcpy-split.cc
    memory-register-async-legacy.cc
    memory-register-async.cc
    memory-register-protection.cc
    memory-touch-async.cc
    memory-touch-register-unregister-async.cc
    memory-unregister-async.cc
    moldability.cc
    sync.cc
    task-dependency-graph.cc
    task-dependency-handle.cc
    task-dependency-interval-matrix.cc
    task-dependency-interval.cc
    task-dependency.cc
    task-format-host.cc
    task-format.cc
    task-gpu-empty.cc
    team-barrier.cc
    team-cpus-master-member.cc
    team-cpus-parallel-for.cc
    team-cpus.cc
    team-device.cc
)

# Loop over each test source file
foreach(TEST_FILE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE ${XKRT})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()

add_subdirectory(api/c/)
