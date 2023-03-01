cmake_minimum_required(VERSION 3.19)

include_directories(${CMAKE_CURRENT_LIST_DIR}/..)
enable_testing()

function(make_test NAME FILE)
    set(TEST_PROJECT "dpm-tests-${NAME}")
    add_executable(${TEST_PROJECT} ${FILE})
    target_link_libraries(${TEST_PROJECT} PRIVATE dpm)

    # Enable max error reporting
    if (MSVC)
        target_compile_options(${TEST_PROJECT} PUBLIC /W3 /WX)
    else ()
        target_compile_options(${TEST_PROJECT} PUBLIC -Wall -Wextra -Wpedantic -Werror -Wno-unknown-pragmas -Wno-logical-not-parentheses -ftemplate-backtrace-limit=0)
    endif ()

    add_test(NAME ${NAME} COMMAND "$<TARGET_FILE:${TEST_PROJECT}>")
endfunction()

project(dpm-tests)

make_test(shuffle ${CMAKE_CURRENT_LIST_DIR}/test_shuffle.cpp)
make_test(blend ${CMAKE_CURRENT_LIST_DIR}/test_blend.cpp)
make_test(frexp ${CMAKE_CURRENT_LIST_DIR}/test_frexp.cpp)
make_test(ilogb ${CMAKE_CURRENT_LIST_DIR}/test_ilogb.cpp)
make_test(trig ${CMAKE_CURRENT_LIST_DIR}/test_trig.cpp)
