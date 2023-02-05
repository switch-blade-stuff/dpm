/*
 * Created by switchblade on 2023-02-05.
 */

#pragma once

#include "mbase.hpp"

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 masksign(__m128 x) noexcept { return _mm_and_ps(x, _mm_set1_ps(-0.0f)); }
		[[nodiscard]] DPM_FORCEINLINE __m128 copysign(__m128 x, __m128 m) noexcept { return _mm_or_ps(abs(x), m); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d masksign(__m128d x) noexcept { return _mm_and_pd(x, _mm_set1_pd(-0.0)); }
		[[nodiscard]] DPM_FORCEINLINE __m128d copysign(__m128d x, __m128d m) noexcept { return _mm_or_pd(abs(x), m); }
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 masksign(__m256 x) noexcept { return _mm256_and_ps(x, _mm256_set1_ps(-0.0f)); }
		[[nodiscard]] DPM_FORCEINLINE __m256 copysign(__m256 x, __m256 m) noexcept { return _mm256_or_ps(abs(x), m); }

		[[nodiscard]] DPM_FORCEINLINE __m256d masksign(__m256d x) noexcept { return _mm256_and_pd(x, _mm256_set1_pd(-0.0)); }
		[[nodiscard]] DPM_FORCEINLINE __m256d copysign(__m256d x, __m256d m) noexcept { return _mm256_or_pd(abs(x), m); }
#endif
	}

	/** Copies sign from elements of vector \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &sign) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto s) { res = detail::copysign(x, detail::masksign(s)); }, result, x, sign);
		return result;
	}
}