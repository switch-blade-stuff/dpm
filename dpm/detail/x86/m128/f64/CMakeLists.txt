
list(APPEND DPM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/utility.hpp
        ${CMAKE_CURRENT_LIST_DIR}/sincos.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp)

list(APPEND DPM_PRIVATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
        ${CMAKE_CURRENT_LIST_DIR}/sincos.cpp)

# Limit allowed instruction sets
if (MSVC)
    set_source_files_properties(
            ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
            ${CMAKE_CURRENT_LIST_DIR}/sincos.cpp
            PROPERTIES COMPILE_FLAGS "/arch:SSE2")
else ()
    set_source_files_properties(
            ${CMAKE_CURRENT_LIST_DIR}/utility.cpp
            ${CMAKE_CURRENT_LIST_DIR}/sincos.cpp
            PROPERTIES COMPILE_FLAGS "-msse2")
endif ()