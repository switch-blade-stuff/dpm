
include(${CMAKE_CURRENT_LIST_DIR}/detail/CMakeLists.txt)

list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/abi.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/flags.hpp
        ${CMAKE_CURRENT_LIST_DIR}/traits.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/math.hpp
        ${CMAKE_CURRENT_LIST_DIR}/simd.hpp)

