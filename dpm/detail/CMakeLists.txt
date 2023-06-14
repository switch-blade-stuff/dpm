
include(${CMAKE_CURRENT_LIST_DIR}/generic/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/x86/CMakeLists.txt)

list(APPEND DPM_HEADERS
        ${CMAKE_CURRENT_LIST_DIR}/arch.hpp
        ${CMAKE_CURRENT_LIST_DIR}/alias.hpp
        ${CMAKE_CURRENT_LIST_DIR}/config.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fconst.hpp
        ${CMAKE_CURRENT_LIST_DIR}/where_expr.hpp)

list(APPEND DPM_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/fconst.cpp
        ${CMAKE_CURRENT_LIST_DIR}/assert.cpp)

