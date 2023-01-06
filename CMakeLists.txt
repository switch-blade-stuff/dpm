cmake_minimum_required(VERSION 3.20)
project(svm)

# Library options
option(SVM_USE_MODULES "Toggles support for C++20 modules" ON)
if (${SVM_USE_MODULES})
    add_compile_definitions(SVM_USE_MODULES)
endif ()

option(SVM_INLINE_EXTENSIONS "Toggles inlining of the library extension namespace" ON)
if (${SVM_INLINE_EXTENSIONS})
    add_compile_definitions(SVM_INLINE_EXTENSIONS)
endif ()

option(SVM_DYNAMIC_DISPATCH "Toggles runtime detection & dispatch of SIMD operations" ON)
if (${SVM_DYNAMIC_DISPATCH})
    add_compile_definitions(SVM_DYNAMIC_DISPATCH)
endif ()

option(SVM_NATIVE_AVX512 "Toggles global support for AVX512 instructions on x86 targets" ON)
if (${SVM_NATIVE_AVX512})
    add_compile_definitions(SVM_NATIVE_AVX512)
endif ()

option(SVM_USE_IPO "Toggles support for inter-procedural optimization" ON)
option(BUILD_SHARED_LIBS "Build SVM as a shared library" OFF)

# Set output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib CACHE STRING "")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE STRING "")

# Include library source subdirectory
set(SVM_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/svm)
include(${SVM_SOURCE_DIR}/CMakeLists.txt)

# Add project targets & include directories
set(SVM_PROJECT ${PROJECT_NAME})
add_library(${SVM_PROJECT}-interface INTERFACE ${SVM_INTERFACE_SOURCES})
add_library(${SVM_PROJECT})

target_sources(${SVM_PROJECT} INTERFACE ${SVM_INTERFACE_SOURCES})
target_sources(${SVM_PROJECT} PRIVATE ${SVM_PRIVATE_SOURCES})
target_sources(${SVM_PROJECT} PUBLIC ${SVM_PUBLIC_SOURCES})

set(SVM_LIBRARY_PATH $<TARGET_FILE:${SVM_PROJECT}>)

# Set symbol visibility
set_target_properties(${SVM_PROJECT} PROPERTIES C_VISIBILITY_PRESET hidden)
set_target_properties(${SVM_PROJECT} PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(${SVM_PROJECT} PROPERTIES VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(${SVM_PROJECT} PROPERTIES DEFINE_SYMBOL "SVM_EXPORT")

# On MSVC, use c++latest instead of c++20 for experimental module support on MSVC
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND SVM_USE_MODULES)
    target_compile_features(${SVM_PROJECT} PUBLIC cxx_std_23)
else ()
    target_compile_features(${SVM_PROJECT} PUBLIC cxx_std_20)
endif ()

# Enable max error reporting
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif ()

# Unit tests
option(SVM_TESTS "Enable unit tests" OFF)
if (${SVM_TESTS})
    include(${CMAKE_CURRENT_LIST_DIR}/test/CMakeLists.txt)
endif ()