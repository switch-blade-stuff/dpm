
include(${CMAKE_CURRENT_LIST_DIR}/m128/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/m256/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/m512/CMakeLists.txt)

list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/abi.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fwd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cpuid.hpp)

list(APPEND DPM_PRIVATE_SOURCES ${CMAKE_CURRENT_LIST_DIR}/cpuid.cpp)

