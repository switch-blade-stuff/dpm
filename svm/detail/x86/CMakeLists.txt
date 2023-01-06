
list(APPEND SVM_INTERFACE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/abi.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type_common.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type_sse.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type_avx.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type_avx512.hpp
        ${CMAKE_CURRENT_LIST_DIR}/type.hpp
        ${CMAKE_CURRENT_LIST_DIR}/cpuid.hpp)

list(APPEND SVM_PRIVATE_SOURCES
        ${CMAKE_CURRENT_LIST_DIR}/cpuid.cpp)

