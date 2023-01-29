/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "utility.hpp"
#include "bitwise.hpp"

namespace dpm::detail
{
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		return std::bit_cast<__m128>(_mm_cmpeq_ps(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		return std::bit_cast<__m128>(_mm_cmpgt_ps(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_lt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		return std::bit_cast<__m128>(_mm_cmplt_ps(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ge(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		return std::bit_cast<__m128>(_mm_cmpge_ps(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_le(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		return std::bit_cast<__m128>(_mm_cmple_ps(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ne(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		return std::bit_cast<__m128>(_mm_cmpneq_ps(va, vb));
	}

#ifdef DPM_HAS_SSE2
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		return std::bit_cast<__m128>(_mm_cmpeq_pd(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		return std::bit_cast<__m128>(_mm_cmpgt_pd(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_lt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		return std::bit_cast<__m128>(_mm_cmplt_pd(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ge(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		return std::bit_cast<__m128>(_mm_cmpge_pd(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_le(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		return std::bit_cast<__m128>(_mm_cmple_pd(va, vb));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ne(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		return std::bit_cast<__m128>(_mm_cmpneq_pd(va, vb));
	}

	/*
	 * Integer comparison availability:
	 *  SSE2: i64 == i64, i32 == i32, i16 == i16, i8 == i8,
	 *        i64 != i64, i32 != i32, i16 != i16, i8 != i8,
	 *        i32 > i32, i16 > i16, i8 > i8,
	 *        u64 == u64, u32 == u32, u16 == u16, u8 == u8,
	 *        u64 != u64, u32 != u32, u16 != u16, u8 != u8
	 *  SSE4.1: i64 > i64
	 */

	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(V a, V b) noexcept requires (sizeof(V) == 16 && (sizeof(T) == 8 || sizeof(T) == 4))
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpeq_epi32(va, vb));
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

#ifdef DPM_HAS_SSE4_1
	template<std::signed_integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpgt_epi64(va, vb));
	}
#endif
	template<std::signed_integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 4)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpgt_epi32(va, vb));
	}
	template<std::signed_integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 2)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpgt_epi16(va, vb));
	}
	template<std::signed_integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 1)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		return std::bit_cast<__m128>(_mm_cmpgt_epi8(va, vb));
	}

	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ne(V a, V b) noexcept requires (sizeof(V) == 16)
	{
		return bit_not(cmp_eq<T>(a, b));
	}
#endif

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
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		return cmp_eq<std::int64_t>(a, b);
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 2)
	{
		return cmp_eq<std::int16_t>(a, b);
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE __m128 mask_eq(V a, V b) noexcept requires (sizeof(V) == 16 && sizeof(T) == 1)
	{
		return cmp_eq<std::int8_t>(a, b);
	}
#endif
}