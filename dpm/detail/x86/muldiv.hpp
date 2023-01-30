/*
 * Created by switchblade on 2023-01-28.
 */

#pragma once

#include "utility.hpp"

namespace dpm::detail
{
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 mul(__m128 a, __m128 b) noexcept { return _mm_mul_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 div(__m128 a, __m128 b) noexcept { return _mm_div_ps(a, b); }

#ifdef DPM_HAS_SSE2
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d mul(__m128d a, __m128d b) noexcept { return _mm_mul_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d div(__m128d a, __m128d b) noexcept { return _mm_div_pd(a, b); }

	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i mul(__m128i a, __m128i b) noexcept { return _mm_mullo_epi16(a, b); }
#endif

#ifdef DPM_HAS_SSE4_1
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i mul(__m128i a, __m128i b) noexcept { return _mm_mullo_epi32(a, b); }
#endif

#ifdef DPM_HAS_AVX
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 mul(__m256 a, __m256 b) noexcept { return _mm256_mul_ps(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d mul(__m256d a, __m256d b) noexcept { return _mm256_mul_pd(a, b); }

	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 div(__m256 a, __m256 b) noexcept { return _mm256_div_ps(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d div(__m256d a, __m256d b) noexcept { return _mm256_div_pd(a, b); }

#ifdef DPM_HAS_AVX2
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i mul(__m256i a, __m256i b) noexcept { return _mm256_mullo_epi16(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i mul(__m256i a, __m256i b) noexcept { return _mm256_mullo_epi32(a, b); }
#else
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i mul(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b) { return mul<T>(a, b); }, a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i mul(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b) { return mul<T>(a, b); }, a, b); }
#endif
#endif

#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512LV)
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i mul(__m128i a, __m128i b) noexcept { return _mm_mullo_epi64(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i mul(__m256i a, __m256i b) noexcept { return _mm256_mullo_epi64(a, b); }
#endif
}