cmake_minimum_required(VERSION 3.20)
project(dpm)

# Library options
#option(DPM_USE_MODULES "Toggles support for C++20 modules" OFF)
#if (${DPM_USE_MODULES})
#    add_compile_definitions(DPM_USE_MODULES)
#endif ()

option(DPM_INLINE_EXTENSIONS "Toggles inlining of the library extension namespace" ON)
if (${DPM_INLINE_EXTENSIONS})
    add_compile_definitions(DPM_INLINE_EXTENSIONS)
endif ()

option(DPM_HANDLE_ERRORS "Toggles detection & reporting of math errors via math_errhandling" ON)
if (${DPM_HANDLE_ERRORS})
    add_compile_definitions(DPM_HANDLE_ERRORS)
endif ()

option(DPM_PROPAGATE_NAN "Toggles guaranteed propagation of NaN" ON)
if (${DPM_PROPAGATE_NAN})
    add_compile_definitions(DPM_PROPAGATE_NAN)
endif ()

option(DPM_USE_SVML "Enables use of math functions provided by SVML" OFF)
if (${DPM_USE_SVML})
    add_compile_definitions(DPM_PROPAGATE_NAN)
endif ()

option(DPM_USE_IPO "Toggles support for inter-procedural optimization" ON)
option(BUILD_SHARED_LIBS "Build DPM as a shared library" OFF)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

# Add project targets & include directories
set(DPM_PROJECT ${PROJECT_NAME})
add_library(${DPM_PROJECT})

# Include library source subdirectory
set(DPM_LIBRARY_PATH $<TARGET_FILE:${DPM_PROJECT}>)
set(DPM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/dpm)
include(${DPM_SOURCE_DIR}/CMakeLists.txt)

# Set symbol visibility
set_target_properties(${DPM_PROJECT} PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(${DPM_PROJECT} PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(${DPM_PROJECT} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(${DPM_PROJECT} PROPERTIES DEFINE_SYMBOL "DPM_EXPORT")

## On MSVC, use c++latest instead of c++20 for experimental module support
#if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND DPM_USE_MODULES)
#    target_compile_features(${DPM_PROJECT} PUBLIC cxx_std_23)
#else ()
#    target_compile_features(${DPM_PROJECT} PUBLIC cxx_std_20)
#endif ()

target_compile_features(${DPM_PROJECT} PUBLIC cxx_std_20)

# Enable max error reporting
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unknown-pragmas -Wno-ignored-attributes)
endif ()

# Unit tests
option(DPM_TESTS "Enable unit tests" OFF)
if (${DPM_TESTS})
    include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
endif ()