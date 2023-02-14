/*
 * Created by switchblade on 2023-02-05.
 */

#pragma once

#include "mbase.hpp"

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 rcp(__m128 x) noexcept { return _mm_rcp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 sqrt(__m128 x) noexcept { return _mm_sqrt_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 rsqrt(__m128 x) noexcept { return _mm_rsqrt_ps(x); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d rcp(__m128d x) noexcept { return _mm_div_pd(_mm_set1_pd(1.0), x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d sqrt(__m128d x) noexcept { return _mm_sqrt_pd(x); }

		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC hypot(__m128 a, __m128 b) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC hypot(__m128d a, __m128d b) noexcept;
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 rcp(__m256 x) noexcept { return _mm256_rcp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 sqrt(__m256 x) noexcept { return _mm256_sqrt_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 rsqrt(__m256 x) noexcept { return _mm256_rsqrt_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d rcp(__m256d x) noexcept { return _mm256_div_pd(_mm256_set1_pd(1.0), x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d sqrt(__m256d x) noexcept { return _mm256_sqrt_pd(x); }

		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC hypot(__m256 a, __m256 b) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC hypot(__m256d a, __m256d b) noexcept;
#endif
	}

	/** Calculates square root of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> sqrt(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::sqrt(x); }, result, x);
		return result;
	}

#ifdef DPM_HAS_SSE2
	/** Calculates square root of the sum of elements in vectors \a a and \a b without causing over or underflow, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> hypot(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::hypot(a, b); }, result, a, b);
		return result;
	}
#endif

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates reciprocal of elements in vector \a x, and returns the resulting vector. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> rcp(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::rcp(x); }, result, x);
			return result;
		}
		/** Calculates reciprocal square root of elements in vector \a x, and returns the resulting vector. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<float, N, A> rsqrt(const detail::x86_simd<float, N, A> &x) noexcept requires detail::x86_overload_any<float, N, A>
		{
			detail::x86_simd<float, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::rsqrt(x); }, result, x);
			return result;
		}
	}
}