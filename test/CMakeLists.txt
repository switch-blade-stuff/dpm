cmake_minimum_required(VERSION 3.19)
project(dpm-tests)
enable_testing()

function(make_test NAME FILE)
    set(TEST_PROJECT "dpm-tests-${NAME}")
    add_executable(${TEST_PROJECT} ${FILE})
    add_test(NAME ${NAME} COMMAND "$<TARGET_FILE:${TEST_PROJECT}>")

    if (DPM_BUILD_SHARED)
        target_link_libraries(${TEST_PROJECT} PRIVATE dpm-shared)
    elseif (DPM_BUILD_STATIC)
        target_link_libraries(${TEST_PROJECT} PRIVATE dpm-static)
    endif ()

    # Enable max error reporting
    target_compile_options(${TEST_PROJECT} PUBLIC ${DPM_COMPILE_OPTIONS})
endfunction()


#make_test(operators ${CMAKE_CURRENT_LIST_DIR}/test_operators.cpp)
make_test(shuffle ${CMAKE_CURRENT_LIST_DIR}/test_shuffle.cpp)
make_test(blend ${CMAKE_CURRENT_LIST_DIR}/test_blend.cpp)
#make_test(copy ${CMAKE_CURRENT_LIST_DIR}/test_copy.cpp)
make_test(fp-manipulation ${CMAKE_CURRENT_LIST_DIR}/test_fmanip.cpp)
make_test(trigonometric ${CMAKE_CURRENT_LIST_DIR}/test_trig.cpp)
make_test(exponential ${CMAKE_CURRENT_LIST_DIR}/test_expon.cpp)
make_test(hypotenuse ${CMAKE_CURRENT_LIST_DIR}/test_hypot.cpp)
