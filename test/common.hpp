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

static inline bool test_err([[maybe_unused]] int except_val, [[maybe_unused]] int errno_val) noexcept
{
	bool result = true;
#if math_errhandling & MATH_ERREXCEPT
	result = result && std::fetestexcept(except_val) == except_val;
#endif
#if math_errhandling & MATH_ERRNO
	result = result && errno == errno_val;
#endif
	return result;
}
static inline void clear_err() noexcept
{
#if math_errhandling & MATH_ERREXCEPT
	std::feclearexcept(FE_ALL_EXCEPT);
#endif
#if math_errhandling & MATH_ERRNO
	errno = 0;
#endif
}

template<typename T>
static inline bool almost_equal(T a, T b, T rel_eps, T eps)
{
	if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))
		return true;

	const auto diff = std::abs(a - b);
	if (diff <= eps) return true;

	a = std::abs(a);
	b = std::abs(b);
	const auto largest = std::max(a, b);
	return diff <= largest * rel_eps;
}
template<typename T>
static inline bool almost_equal(T a, T b, T eps = std::numeric_limits<T>::epsilon()) { return almost_equal(a, b, eps, eps); }