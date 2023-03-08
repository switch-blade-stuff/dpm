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
option(DPM_BUILD_OBJECT "Toggles build of object library target" ON)
option(DPM_BUILD_SHARED "Toggles build of shared library target" ON)
option(DPM_BUILD_STATIC "Toggles build of static library target" ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

# Include library source subdirectory
set(DPM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/dpm)
include(${DPM_SOURCE_DIR}/CMakeLists.txt)

# Add project targets & include directories
if (MSVC)
    set(DPM_COMPILE_OPTIONS /W3 /WX)
else ()
    set(DPM_COMPILE_OPTIONS -Wall -Wextra -Wpedantic -Werror -Wno-unknown-pragmas -Wno-ignored-attributes)
endif ()
if (DPM_BUILD_OBJECT)
    add_library(${PROJECT_NAME}-object OBJECT)
    target_sources(${PROJECT_NAME}-object PUBLIC ${DPM_PUBLIC_SOURCES})
    target_sources(${PROJECT_NAME}-object PRIVATE ${DPM_PRIVATE_SOURCES})
    target_sources(${PROJECT_NAME}-object INTERFACE ${DPM_INTERFACE_SOURCES})
    set_target_properties(${PROJECT_NAME}-object PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

    # Set C++ version
    target_compile_features(${PROJECT_NAME}-object PUBLIC cxx_std_20)

    # Set symbol visibility
    set_target_properties(${PROJECT_NAME}-object PROPERTIES C_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME}-object PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME}-object PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

    # Set shared library definitions
    set_target_properties(${PROJECT_NAME}-object PROPERTIES DEFINE_SYMBOL "DPM_EXPORT")
    target_compile_definitions(${PROJECT_NAME}-object PUBLIC DPM_LIB_SHARED)

    # Enable max error reporting
    target_compile_options(${PROJECT_NAME}-object PRIVATE ${DPM_COMPILE_OPTIONS})
endif ()

if (DPM_BUILD_SHARED)
    add_library(${PROJECT_NAME}-shared SHARED)
    target_sources(${PROJECT_NAME}-shared PUBLIC ${DPM_PUBLIC_SOURCES})
    target_sources(${PROJECT_NAME}-shared PRIVATE ${DPM_PRIVATE_SOURCES})
    target_sources(${PROJECT_NAME}-shared INTERFACE ${DPM_INTERFACE_SOURCES})
    set_target_properties(${PROJECT_NAME}-shared PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

    # Set C++ version
    target_compile_features(${PROJECT_NAME}-shared PUBLIC cxx_std_20)

    # Set symbol visibility
    set_target_properties(${PROJECT_NAME}-shared PROPERTIES C_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME}-shared PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME}-shared PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

    # Set shared library definitions
    set_target_properties(${PROJECT_NAME}-shared PROPERTIES DEFINE_SYMBOL "DPM_EXPORT")
    target_compile_definitions(${PROJECT_NAME}-shared PUBLIC DPM_LIB_SHARED)

    # Enable max error reporting
    target_compile_options(${PROJECT_NAME}-shared PRIVATE ${DPM_COMPILE_OPTIONS})
endif ()

if (DPM_BUILD_STATIC)
    add_library(${PROJECT_NAME}-static STATIC)
    target_sources(${PROJECT_NAME}-static PUBLIC ${DPM_PUBLIC_SOURCES})
    target_sources(${PROJECT_NAME}-static PRIVATE ${DPM_PRIVATE_SOURCES})
    target_sources(${PROJECT_NAME}-static INTERFACE ${DPM_INTERFACE_SOURCES})
    set_target_properties(${PROJECT_NAME}-static PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

    # Set C++ version
    target_compile_features(${PROJECT_NAME}-static PUBLIC cxx_std_20)

    # Set symbol visibility
    set_target_properties(${PROJECT_NAME}-static PROPERTIES C_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME}-static PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${PROJECT_NAME}-static PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

    # Set shared library definitions
    set_target_properties(${PROJECT_NAME}-static PROPERTIES DEFINE_SYMBOL "DPM_EXPORT")
    target_compile_definitions(${PROJECT_NAME}-static PUBLIC DPM_LIB_STATIC)

    # Enable max error reporting
    target_compile_options(${PROJECT_NAME}-static PRIVATE ${DPM_COMPILE_OPTIONS})
endif ()

# Unit tests
option(DPM_TESTS "Enable unit tests" OFF)
if (${DPM_TESTS})
    include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
endif ()
