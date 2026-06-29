# Tests for the K-dimensional "lp-tree" data structure used by the BLAS /
# interval dependency trees.

# Automated, non-interactive insertion test.
xkrt_add_test(NAME khp-tree-random SOURCE random.cc LABELS data-structure khp-tree)

# The interactive drivers read hyper-rectangles from stdin; they are built for
# compile-coverage and manual experimentation, but are NOT registered as tests.
foreach(_interactive interactive-2D)
    add_executable(khp-tree-${_interactive} ${_interactive}.cc)
    target_link_libraries(khp-tree-${_interactive} PRIVATE ${XKRT})
    target_include_directories(khp-tree-${_interactive} PRIVATE ${XKRT_TESTS_ROOT})
    target_compile_options(khp-tree-${_interactive} PRIVATE -UNDEBUG -g)
endforeach()
