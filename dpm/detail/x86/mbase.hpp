/*
 * Created by switchblade on 2023-02-01.
 */

#pragma once

#include "math_fwd.hpp"
#include "type.hpp"

#ifdef DPM_HAS_SSE2

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC("sse2") fmod(__m128 a, __m128 b) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC("sse2") fmod(__m128d a, __m128d b) noexcept;
	}

	/** Calculates floating-point remainder of elements in \a a divided by elements in vector \a b.
	 * @note The current SIMD implementation of `fmod` does not handle subnormal elements of \a a or \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmod(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::fmod(a, b); }, result, a, b);
		return result;
	}
}

#endif