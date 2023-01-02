
include(${CMAKE_CURRENT_LIST_DIR}/detail/CMakeLists.txt)

list(APPEND SVM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/abi.hpp
        ${CMAKE_CURRENT_LIST_DIR}/simd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/flags.hpp
        ${CMAKE_CURRENT_LIST_DIR}/traits.hpp)

