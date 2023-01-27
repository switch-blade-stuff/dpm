/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "utility.hpp"

namespace dpm::detail
{
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
#ifdef DPM_HAS_SSE2
		if constexpr (std::integral<T>)
		{
			const auto va = std::bit_cast<__m128i>(a);
			const auto vb = std::bit_cast<__m128i>(b);
			return std::bit_cast<__m128>(_mm_cmpeq_epi32(va, vb));
		}
		else
#endif
		{
			const auto va = std::bit_cast<__m128>(a);
			const auto vb = std::bit_cast<__m128>(b);
			return std::bit_cast<__m128>(_mm_cmpeq_ps(va, vb));
		}
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
#ifndef DPM_HAS_SSE2
		return cmp_eq<T>(a, bit_xor(b, fill<V>(std::bit_cast<T>(0x3fff'ffff))));
#else
		return cmp_eq<std::int32_t>(a, b);
#endif
	}

#ifdef DPM_HAS_SSE2
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		if constexpr (std::integral<T>)
		{
			const auto va = std::bit_cast<__m128i>(a);
			const auto vb = std::bit_cast<__m128i>(b);
			return std::bit_cast<__m128>(_mm_cmpeq_epi32(va, vb));
		}
		else
		{
			const auto va = std::bit_cast<__m128d>(a);
			const auto vb = std::bit_cast<__m128d>(b);
			return _mm_cmpeq_pd(va, vb);
		}
	}
	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 2)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpeq_epi16(va, vb));
	}
	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 1)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpeq_epi8(va, vb));
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		return cmp_eq<std::int64_t>(a, b);
	}
	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 2)
	{
		return cmp_eq<std::int16_t>(a, b);
	}
	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 1)
	{
		return cmp_eq<std::int8_t>(a, b);
	}
#endif
}