
include(${CMAKE_CURRENT_LIST_DIR}/sse/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/avx/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/avx512/CMakeLists.txt)

list(APPEND SVM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/abi.hpp
        ${CMAKE_CURRENT_LIST_DIR}/common.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cpuid.hpp)

list(APPEND SVM_PRIVATE_SOURCES ${CMAKE_CURRENT_LIST_DIR}/cpuid.cpp)

