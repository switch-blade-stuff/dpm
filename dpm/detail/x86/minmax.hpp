/*
 * Created by switchblade on 2023-01-30.
 */

#pragma once

#include "utility.hpp"

namespace dpm::detail
{
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 min(__m128 a, __m128 b) noexcept { return _mm_min_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 max(__m128 a, __m128 b) noexcept { return _mm_max_ps(a, b); }

#ifdef DPM_HAS_SSE2
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d min(__m128d a, __m128d b) noexcept { return _mm_min_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d max(__m128d a, __m128d b) noexcept { return _mm_max_pd(a, b); }

	template<unsigned_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epu8(a, b); }
	template<unsigned_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epu8(a, b); }
	template<signed_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epi16(a, b); }
	template<signed_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epi16(a, b); }
#endif

#ifdef DPM_HAS_SSE4_1
	template<signed_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epi8(a, b); }
	template<signed_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epi8(a, b); }
	template<unsigned_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epu16(a, b); }
	template<unsigned_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epu16(a, b); }
	template<signed_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epi32(a, b); }
	template<signed_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epi32(a, b); }
	template<unsigned_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epu32(a, b); }
	template<unsigned_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epu32(a, b); }
#endif

#ifdef DPM_HAS_AVX
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 min(__m256 a, __m256 b) noexcept { return _mm256_min_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 max(__m256 a, __m256 b) noexcept { return _mm256_max_ps(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d min(__m256d a, __m256d b) noexcept { return _mm256_min_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d max(__m256d a, __m256d b) noexcept { return _mm256_max_pd(a, b); }
#endif

#ifdef DPM_HAS_AVX2
	template<signed_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epi8(a, b); }
	template<signed_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epi8(a, b); }
	template<unsigned_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epu8(a, b); }
	template<unsigned_integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epu8(a, b); }
	template<signed_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epi16(a, b); }
	template<signed_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epi16(a, b); }
	template<unsigned_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epu16(a, b); }
	template<unsigned_integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epu16(a, b); }
	template<signed_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epi32(a, b); }
	template<signed_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epi32(a, b); }
	template<unsigned_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epu32(a, b); }
	template<unsigned_integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epu32(a, b); }
#endif

#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
	template<signed_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epi64(a, b); }
	template<signed_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epi64(a, b); }
	template<unsigned_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i min(__m128i a, __m128i b) noexcept { return _mm_min_epu64(a, b); }
	template<unsigned_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i max(__m128i a, __m128i b) noexcept { return _mm_max_epu64(a, b); }
	template<signed_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epi64(a, b); }
	template<signed_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epi64(a, b); }
	template<unsigned_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i min(__m256i a, __m256i b) noexcept { return _mm256_min_epu64(a, b); }
	template<unsigned_integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i max(__m256i a, __m256i b) noexcept { return _mm256_max_epu64(a, b); }
#endif
}