/*
 * Created by switchblade on 2023-01-13.
 */

#pragma once

#include "../../../define.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include <immintrin.h>

#ifndef DPM_USE_IMPORT

#include <span>

#endif

namespace dpm::detail
{
	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] inline __m128d DPM_FORCEINLINE DPM_TARGET("fma") x86_polevl_f64_fma(__m128d x, __m128d y, std::span<const double, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = _mm_fmadd_pd(y, x, _mm_set1_pd(c[J]));
			return x86_polevl_f64_fma<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] inline __m128d DPM_FORCEINLINE DPM_TARGET("fma") x86_polevl_f64_fma(__m128d x, std::span<const double, N> c) noexcept
	{
		return x86_polevl_f64_fma<N, N>(x, _mm_set1_pd(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_polevl_f64_sse(__m128d x, __m128d y, std::span<const double, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = _mm_add_pd(_mm_mul_pd(y, x), _mm_set1_pd(c[J]));
			return x86_polevl_f64_sse<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_polevl_f64_sse(__m128d x, std::span<const double, N> c) noexcept
	{
		return x86_polevl_f64_sse<N, N>(x, _mm_set1_pd(c[0]), c);
	}
}

#endif