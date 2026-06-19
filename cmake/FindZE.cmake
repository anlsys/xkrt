# Try to find ZE headers and libraries.
#
# Usage of this module:
#     find_package(ZE REQUIRED)
#
# Variables used by this module (can be set as CMake variables or Environment variables):
#  ZE_ROOT or ZE_PREFIX   Set this to the root installation of Intel Level Zero 
#                         if the module has problems finding the path automatically.
#
# Variables defined by this module:
#  ZE_FOUND               System has ZE libraries and headers
#  ZE_LIBRARIES           The ZE library for Runtime API
#  ZE_INCLUDE_DIRS        The location of ZE headers
#
# Targets defined:
#  ZE::ZE                 Modern imported target to link against

# 1. Gather hint paths from both CMake scope and your active shell environment
set(ZE_HINTS
    ${ZE_ROOT}
    ${ZE_PREFIX}
    $ENV{ZE_ROOT}
    $ENV{ZE_PREFIX}
    $ENV{ZE_PATH}
    $ENV{ONEAPI_ROOT}/level-zero/latest
    $ENV{ONEAPI_ROOT}
)

# 2. Search for headers (handles both direct include/ and subfolder include/level_zero/)
find_path(ZE_INCLUDE_DIRS
    NAMES level_zero/ze_api.h ze_api.h
    HINTS ${ZE_HINTS}
    PATH_SUFFIXES include
)

# 3. Search for the loader library
find_library(ZE_LIBRARIES
    NAMES ze_loader
    HINTS ${ZE_HINTS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

# 4. Handle the standard REQUIRED / FOUND arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZE DEFAULT_MSG
    ZE_LIBRARIES
    ZE_INCLUDE_DIRS
)

# 5. Create a modern imported target so consumers don't have to manage raw paths
if(ZE_FOUND AND NOT TARGET ZE::ZE)
    add_library(ZE::ZE UNKNOWN IMPORTED)
    set_target_properties(ZE::ZE PROPERTIES
        IMPORTED_LOCATION "${ZE_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ZE_INCLUDE_DIRS}"
    )
endif()

mark_as_advanced(
    ZE_LIBRARIES
    ZE_INCLUDE_DIRS
)
