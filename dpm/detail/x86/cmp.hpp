/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "bitwise.hpp"

namespace dpm::detail
{
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_eq(__m128 a, __m128 b) noexcept { return _mm_cmpeq_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_gt(__m128 a, __m128 b) noexcept { return _mm_cmpgt_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_lt(__m128 a, __m128 b) noexcept { return _mm_cmplt_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ge(__m128 a, __m128 b) noexcept { return _mm_cmpge_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_le(__m128 a, __m128 b) noexcept { return _mm_cmple_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 cmp_ne(__m128 a, __m128 b) noexcept { return _mm_cmpneq_ps(a, b); }

#ifdef DPM_HAS_SSE2
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d cmp_eq(__m128d a, __m128d b) noexcept { return _mm_cmpeq_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d cmp_gt(__m128d a, __m128d b) noexcept { return _mm_cmpgt_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d cmp_lt(__m128d a, __m128d b) noexcept { return _mm_cmplt_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d cmp_ge(__m128d a, __m128d b) noexcept { return _mm_cmpge_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d cmp_le(__m128d a, __m128d b) noexcept { return _mm_cmple_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d cmp_ne(__m128d a, __m128d b) noexcept { return _mm_cmpneq_pd(a, b); }

	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_eq(__m128i a, __m128i b) noexcept { return _mm_cmpeq_epi8(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_eq(__m128i a, __m128i b) noexcept { return _mm_cmpeq_epi16(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_eq(__m128i a, __m128i b) noexcept { return _mm_cmpeq_epi32(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_eq(__m128i a, __m128i b) noexcept
	{
#ifdef DPM_HAS_SSE4_1
		return _mm_cmpeq_epi64(a, b);
#else
		const auto tmp = _mm_cmpeq_epi32(a, b);
		return _mm_and_si128(tmp, _mm_slli_epi64(tmp, 32));
#endif
	}

	template<signed_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_gt(__m128i a, __m128i b) noexcept { return _mm_cmpgt_epi8(a, b); }
	template<signed_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_gt(__m128i a, __m128i b) noexcept { return _mm_cmpgt_epi16(a, b); }
	template<signed_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_gt(__m128i a, __m128i b) noexcept { return _mm_cmpgt_epi32(a, b); }
#ifdef DPM_HAS_SSE4_1
	template<signed_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i cmp_gt(__m128i a, __m128i b) noexcept { return _mm_cmpgt_epi64(a, b); }
#endif
#endif

#ifdef DPM_HAS_AVX
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 cmp_eq(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_EQ_OQ); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 cmp_gt(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 cmp_lt(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 cmp_ge(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_GE_OQ); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 cmp_le(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_LE_OQ); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 cmp_ne(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ); }

	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d cmp_eq(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_EQ_OQ); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d cmp_gt(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_GT_OQ); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d cmp_lt(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_LT_OQ); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d cmp_ge(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_GE_OQ); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d cmp_le(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_LE_OQ); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d cmp_ne(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_NEQ_OQ); }

#ifdef DPM_HAS_AVX2
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_eq(__m256i a, __m256i b) noexcept { return _mm256_cmpeq_epi8(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_eq(__m256i a, __m256i b) noexcept { return _mm256_cmpeq_epi16(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_eq(__m256i a, __m256i b) noexcept { return _mm256_cmpeq_epi32(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_eq(__m256i a, __m256i b) noexcept { return _mm256_cmpeq_epi64(a, b); }

	template<signed_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_gt(__m256i a, __m256i b) noexcept { return _mm256_cmpgt_epi8(a, b); }
	template<signed_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_gt(__m256i a, __m256i b) noexcept { return _mm256_cmpgt_epi16(a, b); }
	template<signed_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_gt(__m256i a, __m256i b) noexcept { return _mm256_cmpgt_epi32(a, b); }
	template<signed_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_gt(__m256i a, __m256i b) noexcept { return _mm256_cmpgt_epi64(a, b); }
#else
	template<std::integral T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_eq(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b) { return cmp_eq<T>(a, b); }, a, b); }
	template<std::integral T>
	[[nodiscard]] DPM_FORCEINLINE __m256i cmp_gt(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b) { return cmp_gt<T>(a, b); }, a, b); }
#endif
#endif

	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V cmp_ne(V a, V b) noexcept { return bit_not(cmp_eq<T>(a, b)); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V mask_eq(V a, V b) noexcept requires (sizeof(T) == 4)
	{
#ifndef DPM_HAS_SSE2
		return cmp_eq<float>(a, bit_xor(b, fill<V>(std::bit_cast<float>(0x3fff'ffff))));
#else
		using ivec_t = select_vector_t<std::int32_t, sizeof(V)>;
		const auto ai = std::bit_cast<ivec_t>(a);
		const auto bi = std::bit_cast<ivec_t>(b);
		return std::bit_cast<V>(cmp_eq<std::int32_t>(ai, bi));
#endif
	}
#ifdef DPM_HAS_SSE2
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V mask_eq(V a, V b) noexcept requires (sizeof(T) == 8)
	{
		using ivec_t = select_vector_t<std::int64_t, sizeof(V)>;
		const auto ai = std::bit_cast<ivec_t>(a);
		const auto bi = std::bit_cast<ivec_t>(b);
		return std::bit_cast<V>(cmp_eq<std::int64_t>(ai, bi));
	}
	template<integral_of_size<2> T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V mask_eq(V a, V b) noexcept { return cmp_eq<std::int16_t>(a, b); }
	template<integral_of_size<1> T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V mask_eq(V a, V b) noexcept { return cmp_eq<std::int8_t>(a, b); }
#endif
}