/*
 * Created by switchblade on 2023-02-27.
 */

#pragma once

/* Always test with error handling on. Otherwise, error conditions are UB. */
#ifndef DPM_HANDLE_ERRORS
#define DPM_HANDLE_ERRORS
#endif

#include <dpm/simd.hpp>

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#define TEST_ASSERT(x) assert(x)

template<typename T>
static inline bool almost_equal(T a, T b, T rel_eps, T eps)
{
	if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b))
		return true;
	if (std::isnan(a) && std::isnan(b))
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