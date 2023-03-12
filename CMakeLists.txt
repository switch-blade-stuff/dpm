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

option(DPM_BUILD_SHARED "Toggles build of shared library target" ON)
option(DPM_BUILD_STATIC "Toggles build of static library target" ON)
option(DPM_USE_IPO "Toggles support for inter-procedural optimization" ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

# Include library source subdirectory
set(DPM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/dpm)
include(${DPM_SOURCE_DIR}/CMakeLists.txt)

# Add shared & static library targets
function(dpm_configure_target NAME TYPE)
    add_library(${NAME} ${TYPE})
    target_sources(${NAME} PUBLIC ${DPM_PUBLIC_SOURCES})
    target_sources(${NAME} PRIVATE ${DPM_PRIVATE_SOURCES})
    target_sources(${NAME} INTERFACE ${DPM_INTERFACE_SOURCES})
    target_include_directories(${NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(${NAME} PROPERTIES OUTPUT_NAME "${PROJECT_NAME}")

    # Set C++ version
    target_compile_features(${NAME} PUBLIC cxx_std_20)

    # Set symbol visibility
    set_target_properties(${NAME} PROPERTIES C_VISIBILITY_PRESET hidden)
    set_target_properties(${NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    set_target_properties(${NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

    # Set shared library definitions
    set_target_properties(${NAME} PROPERTIES DEFINE_SYMBOL "DPM_EXPORT")

    # Enable max error reporting
    if (MSVC)
        target_compile_options(${NAME} PRIVATE /W3 /WX)
    else ()
        target_compile_options(${NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-unknown-pragmas -Wno-ignored-attributes)
    endif ()
endfunction()
if (DPM_BUILD_SHARED)
    dpm_configure_target(${PROJECT_NAME}-shared SHARED)
endif ()
if (DPM_BUILD_STATIC)
    dpm_configure_target(${PROJECT_NAME}-static STATIC)
    set_target_properties(${PROJECT_NAME}-static PROPERTIES PREFIX "lib")
    target_compile_definitions(${PROJECT_NAME}-static PUBLIC DPM_LIB_STATIC)
endif ()

# Unit tests
option(DPM_TESTS "Enable unit tests" OFF)
if (${DPM_TESTS})
    include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
endif ()
