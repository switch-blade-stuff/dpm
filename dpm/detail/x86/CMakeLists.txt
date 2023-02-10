
list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/cpuid.hpp
        ${CMAKE_CURRENT_LIST_DIR}/abi.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type_fwd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/transform.hpp
        ${CMAKE_CURRENT_LIST_DIR}/bitwise.hpp
        ${CMAKE_CURRENT_LIST_DIR}/addsub.hpp
        ${CMAKE_CURRENT_LIST_DIR}/muldiv.hpp
        ${CMAKE_CURRENT_LIST_DIR}/minmax.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cmp.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cvt.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/math_fwd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/mbase.hpp
        ${CMAKE_CURRENT_LIST_DIR}/power.hpp
        ${CMAKE_CURRENT_LIST_DIR}/class.hpp
        ${CMAKE_CURRENT_LIST_DIR}/round.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fmanip.hpp
        ${CMAKE_CURRENT_LIST_DIR}/polevl.hpp
        ${CMAKE_CURRENT_LIST_DIR}/trig.hpp
        ${CMAKE_CURRENT_LIST_DIR}/math.hpp)

list(APPEND DPM_PRIVATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/cpuid.cpp
        ${CMAKE_CURRENT_LIST_DIR}/sincos.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tan.cpp
        ${CMAKE_CURRENT_LIST_DIR}/asincos.cpp
        ${CMAKE_CURRENT_LIST_DIR}/atan.cpp)
