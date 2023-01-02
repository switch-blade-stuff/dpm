
include(${CMAKE_CURRENT_LIST_DIR}/generic/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/arm/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/x86/CMakeLists.txt)

list(APPEND SVM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/arch.hpp
        ${CMAKE_CURRENT_LIST_DIR}/define.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/dispatch.hpp
        ${CMAKE_CURRENT_LIST_DIR}/reference.hpp)

