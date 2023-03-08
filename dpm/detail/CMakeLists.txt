
include(${CMAKE_CURRENT_LIST_DIR}/generic/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/x86/CMakeLists.txt)

list(APPEND DPM_INTERFACE_SOURCES ${CMAKE_CURRENT_LIST_DIR}/where_expr.hpp)

list(APPEND DPM_PUBLIC_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/api.hpp
        ${CMAKE_CURRENT_LIST_DIR}/arch.hpp
        ${CMAKE_CURRENT_LIST_DIR}/alias.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/define.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fconst.hpp)

list(APPEND DPM_PRIVATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/fconst.cpp
        ${CMAKE_CURRENT_LIST_DIR}/assert.cpp)

