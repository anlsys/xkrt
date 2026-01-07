# Try to find ROCm SMI headers and libraries.
#
# Usage:
#     find_package(RSMI)
#
# Variables that can be set before calling find_package:
#
#  RSMI_PREFIX        Root installation of ROCm (default: /opt/rocm)
#
# Variables defined by this module:
#
#  RSMI_FOUND         System has ROCm SMI
#  RSMI_LIBRARIES     The ROCm SMI library
#  RSMI_INCLUDE_DIRS  The location of ROCm SMI headers

# Default prefix
if(NOT RSMI_PREFIX)
    set(RSMI_PREFIX /opt/rocm)
endif()

# Find header
find_path(RSMI_INCLUDE_DIRS
    NAMES rocm_smi/rocm_smi.h
    HINTS
        ${RSMI_PREFIX}/include
        $ENV{ROCM_PATH}/include
)

# Find library
find_library(RSMI_LIBRARIES
    NAMES rocm_smi64
    HINTS
        ${RSMI_PREFIX}/lib
        ${RSMI_PREFIX}/lib64
        $ENV{ROCM_PATH}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RSMI DEFAULT_MSG
    RSMI_LIBRARIES
    RSMI_INCLUDE_DIRS
)

mark_as_advanced(
    RSMI_PREFIX
    RSMI_LIBRARIES
    RSMI_INCLUDE_DIRS
)

