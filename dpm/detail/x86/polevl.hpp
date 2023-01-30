/*
 * Created by switchblade on 2023-01-13.
 */

#pragma once

#include "fmadd.hpp"

#ifndef DPM_USE_IMPORT

#include <span>

#endif

namespace dpm::detail
{
	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m128 DPM_TARGET("fma") polevl_fma(__m128 x, __m128 y, std::span<const float, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = _mm_fmadd_ps(y, x, _mm_set1_ps(c[J]));
			return polevl_fma<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m128 DPM_TARGET("fma") polevl_fma(__m128 x, std::span<const float, N> c) noexcept
	{
		return polevl_fma<N, N>(x, _mm_set1_pd(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m128 polevl_sse(__m128 x, __m128 y, std::span<const float, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = fmadd_sse(y, x, _mm_set1_ps(c[J]));
			return polevl_sse<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m128 polevl_sse(__m128 x, std::span<const float, N> c) noexcept
	{
		return polevl_sse<N, N>(x, _mm_set1_pd(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m128d DPM_TARGET("fma") polevl_fma(__m128d x, __m128d y, std::span<const double, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = _mm_fmadd_pd(y, x, _mm_set1_pd(c[J]));
			return polevl_fma<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m128d DPM_TARGET("fma") polevl_fma(__m128d x, std::span<const double, N> c) noexcept
	{
		return polevl_fma<N, N>(x, _mm_set1_pd(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m128d polevl_sse(__m128d x, __m128d y, std::span<const double, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = fmadd_sse(y, x, _mm_set1_pd(c[J]));
			return polevl_sse<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m128d polevl_sse(__m128d x, std::span<const double, N> c) noexcept
	{
		return polevl_sse<N, N>(x, _mm_set1_pd(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m256 DPM_TARGET("fma") polevl_fma(__m256 x, __m256 y, std::span<const float, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(c[J]));
			return polevl_fma<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m256 DPM_TARGET("fma") polevl_fma(__m256 x, std::span<const float, N> c) noexcept
	{
		return polevl_fma<N, N>(x, _mm256_set1_ps(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m256 DPM_TARGET("avx") polevl_avx(__m256 x, __m256 y, std::span<const float, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = fmadd_avx(y, x, _mm256_set1_ps(c[J]));
			return polevl_avx<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m256 DPM_TARGET("avx") polevl_avx(__m256 x, std::span<const float, N> c) noexcept
	{
		return polevl_avx<N, N>(x, _mm256_set1_ps(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m256d DPM_TARGET("fma") polevl_fma(__m256d x, __m256d y, std::span<const double, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = _mm256_fmadd_pd(y, x, _mm256_set1_pd(c[J]));
			return polevl_fma<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m256d DPM_TARGET("fma") polevl_fma(__m256d x, std::span<const double, N> c) noexcept
	{
		return polevl_fma<N, N>(x, _mm256_set1_pd(c[0]), c);
	}

	template<std::size_t N, std::size_t I, std::size_t J = 0>
	[[nodiscard]] DPM_FORCEINLINE __m256d DPM_TARGET("avx") polevl_avx(__m256d x, __m256d y, std::span<const double, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = fmadd_avx(y, x, _mm256_set1_pd(c[J]));
			return polevl_avx<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE __m256d DPM_TARGET("avx") polevl_avx(__m256d x, std::span<const double, N> c) noexcept
	{
		return polevl_avx<N, N>(x, _mm256_set1_pd(c[0]), c);
	}
}