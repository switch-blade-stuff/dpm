
include(${CMAKE_CURRENT_LIST_DIR}/i8/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/i16/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/i32/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/i64/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/f32/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/f64/CMakeLists.txt)

list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/bitwise.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cmp.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cvt.hpp
        ${CMAKE_CURRENT_LIST_DIR}/mask.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/math.hpp)