# CMake Minimum version
cmake_minimum_required(VERSION 3.17)
include(CTest)
enable_testing()

# Project name
project(tests)

###############
# C++ version #
###############
set(CMAKE_CXX_STANDARD 20) 
set(CMAKE_CXX_STANDARD_REQUIRED ON) 
set(CMAKE_CXX_EXTENSIONS OFF)  # Optional: disables compiler-specific extensions

# Root of the test-suite, used to resolve `#include <common/skip.h>` from tests.
set(XKRT_TESTS_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# Tests rely on `assert(...)` for their checks. `assert` is disabled when NDEBUG
# is defined (e.g. with `-DCMAKE_BUILD_TYPE=Release`), which would make tests
# pass trivially. We therefore *force assertions on* for every test target,
# independently of the build type, by undefining NDEBUG on the test sources.
# Tests are still expected to be run against a Debug build of the library.
if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS
        "[tests] CMAKE_BUILD_TYPE is '${CMAKE_BUILD_TYPE}', not 'Debug'. "
        "Test assertions are force-enabled (-UNDEBUG) so the suite stays "
        "meaningful, but building tests in Debug is recommended.")
endif()

#
# xkrt_add_test(NAME <name> SOURCE <file> [GPU] [LABELS <label>...])
#
#   NAME    unique test/target name
#   SOURCE  test source file (.cc or .c)
#   GPU     mark the test as requiring a GPU: it is given the `gpu` label and
#           its `SKIP_RETURN_CODE` is set so that a test which detects no GPU
#           (see common/skip.h, returns 77) is reported as "Skipped" rather
#           than failing. Tests without GPU get the `host` label.
#   LABELS  extra CTest labels (e.g. an area: task, memory, team, ...)
#
function(xkrt_add_test)
    cmake_parse_arguments(T "GPU" "NAME;SOURCE" "LABELS" ${ARGN})

    if (NOT T_NAME OR NOT T_SOURCE)
        message(FATAL_ERROR "xkrt_add_test requires NAME and SOURCE")
    endif()

    add_executable(${T_NAME} ${T_SOURCE})
    target_link_libraries(${T_NAME} PRIVATE ${XKRT})
    target_include_directories(${T_NAME} PRIVATE ${XKRT_TESTS_ROOT})

    # always keep assertions active in tests
    target_compile_options(${T_NAME} PRIVATE -UNDEBUG -g)

    add_test(NAME ${T_NAME} COMMAND ${T_NAME})

    set(_labels ${T_LABELS})
    if (T_GPU)
        list(APPEND _labels gpu)
        set_tests_properties(${T_NAME} PROPERTIES SKIP_RETURN_CODE 77)
    else()
        list(APPEND _labels host)
    endif()

    set_tests_properties(${T_NAME} PROPERTIES
        LABELS "${_labels}"
        TIMEOUT 600)
endfunction()

# Feature areas
add_subdirectory(lifecycle)
add_subdirectory(task)
add_subdirectory(memory)
add_subdirectory(team)
add_subdirectory(file)
add_subdirectory(api/c)
add_subdirectory(khp-tree)
