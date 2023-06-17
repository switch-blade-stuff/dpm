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

option(BUILD_SHARED_LIBS "Toggles build as a shared library" ON)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

# Add library target
add_library(${PROJECT_NAME})

# Include library source subdirectory
set(DPM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/dpm)
include(${DPM_SOURCE_DIR}/CMakeLists.txt)

# Generate CMake export header to be included by config.hpp
include(GenerateExportHeader)
set(DPM_EXPORT_HEADER ${DPM_SOURCE_DIR}/detail/export.gen.hpp)
generate_export_header(${PROJECT_NAME} BASE_NAME DPM EXPORT_FILE_NAME ${DPM_EXPORT_HEADER})
list(APPEND DPM_HEADERS ${DPM_EXPORT_HEADER})

# Add sources & includes
target_sources(${PROJECT_NAME} PUBLIC ${DPM_HEADERS})
target_sources(${PROJECT_NAME} PRIVATE ${DPM_SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# Set C++ version
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Set symbol visibility
set_target_properties(${PROJECT_NAME} PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(${PROJECT_NAME} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)

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
