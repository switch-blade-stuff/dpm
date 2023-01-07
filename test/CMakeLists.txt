cmake_minimum_required(VERSION 3.19)

include_directories(${CMAKE_CURRENT_LIST_DIR}/..)
enable_testing()

project(svm-tests)

add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_LIST_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE svm)

# Enable max error reporting
if (MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
else ()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif ()

# On MSVC, use c++latest instead of c++20 for experimental module support
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND SVM_USE_MODULES)
    target_compile_features(${SVM_PROJECT} PUBLIC cxx_std_23)
else ()
    target_compile_features(${SVM_PROJECT} PUBLIC cxx_std_20)
endif ()