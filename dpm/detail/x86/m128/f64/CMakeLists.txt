
list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/polevl.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fmadd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/class.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sign.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp)


# Limit allowed instruction sets
if (MSVC)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/sincos.cpp PROPERTIES COMPILE_FLAGS "/arch:SSE2")
else ()
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/sincos.cpp PROPERTIES COMPILE_FLAGS "-msse2 -Wno-uninitialized")
endif ()