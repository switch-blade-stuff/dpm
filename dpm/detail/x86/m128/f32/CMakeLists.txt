
list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sincos.hpp
        ${CMAKE_CURRENT_LIST_DIR}/fmadd.hpp
        ${CMAKE_CURRENT_LIST_DIR}/class.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sign.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp)

list(APPEND DPM_PRIVATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/sincos.cpp)

# Limit allowed instruction sets
if (MSVC)
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/sincos.cpp PROPERTIES COMPILE_FLAGS "/arch:SSE")
else ()
    set_source_files_properties(${CMAKE_CURRENT_LIST_DIR}/sincos.cpp PROPERTIES COMPILE_FLAGS "-msse")
endif ()