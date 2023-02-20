
include(${CMAKE_CURRENT_LIST_DIR}/generic/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/x86/CMakeLists.txt)

target_sources(${DPM_PROJECT} INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/api.hpp
        ${CMAKE_CURRENT_LIST_DIR}/arch.hpp
        ${CMAKE_CURRENT_LIST_DIR}/define.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/alias.hpp
        ${CMAKE_CURRENT_LIST_DIR}/where_expr.hpp)

target_sources(${DPM_PROJECT} PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/fconst.hpp)

target_sources(${DPM_PROJECT} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/fconst.cpp
        ${CMAKE_CURRENT_LIST_DIR}/assert.cpp)

