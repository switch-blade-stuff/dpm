/*
 * Created by switchblade on 2023-01-11.
 */

#pragma once

#include "sign.hpp"

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 isnan(__m128 x) noexcept { return _mm_cmpunord_ps(x, x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 isinf(__m128 x) noexcept
		{
			const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
			return _mm_cmpeq_ps(abs(x), inf);
		}

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d isnan(__m128d x) noexcept { return _mm_cmpunord_pd(x, x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d isinf(__m128d x) noexcept
		{
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
			return _mm_cmpeq_pd(abs(x), inf);
		}
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 isnan(__m256 x) noexcept { return mux_128x2<__m256>([](auto x) { return isnan(x); }, x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d isnan(__m256d x) noexcept { return mux_128x2<__m256d>([](auto x) { return isnan(x); }, x); }

		[[nodiscard]] DPM_FORCEINLINE __m256 isinf(__m256 x) noexcept
		{
			const auto inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
			return cmp_eq<float>(abs(x), inf);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d isinf(__m256d x) noexcept
		{
			const auto inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
			return cmp_eq<double>(abs(x), inf);
		}
#endif
	}

	/** Determines is elements of \a x are unordered NaN and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isnan(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::isnan(x); }, result, x);
		return result;
	}
	/** Determines is elements of \a x are infinite and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isinf(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::isinf(x); }, result, x);
		return result;
	}
}