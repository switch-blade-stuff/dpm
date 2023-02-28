/*
 * Created by switchblade on 2023-02-27.
 */

#pragma once

/* Always test with error handling on. Otherwise, error conditions are UB. */
#ifndef DPM_HANDLE_ERRORS
#define DPM_HANDLE_ERRORS
#endif

#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif

#include <dpm/simd.hpp>

#define TEST_ASSERT(x) DPM_ASSERT_ALWAYS(x)