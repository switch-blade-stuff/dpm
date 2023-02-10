/*
 * Created by switchblade on 2023-01-28.
 */

#pragma once

#include "bitwise.hpp"

namespace dpm::detail
{
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 add(__m128 a, __m128 b) noexcept { return _mm_add_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m128 sub(__m128 a, __m128 b) noexcept { return _mm_sub_ps(a, b); }

#ifdef DPM_HAS_SSE2
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d add(__m128d a, __m128d b) noexcept { return _mm_add_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m128d sub(__m128d a, __m128d b) noexcept { return _mm_sub_pd(a, b); }

	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i add(__m128i a, __m128i b) noexcept { return _mm_add_epi8(a, b); }
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i sub(__m128i a, __m128i b) noexcept { return _mm_sub_epi8(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i add(__m128i a, __m128i b) noexcept { return _mm_add_epi16(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i sub(__m128i a, __m128i b) noexcept { return _mm_sub_epi16(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i add(__m128i a, __m128i b) noexcept { return _mm_add_epi32(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i sub(__m128i a, __m128i b) noexcept { return _mm_sub_epi32(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i add(__m128i a, __m128i b) noexcept { return _mm_add_epi64(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i sub(__m128i a, __m128i b) noexcept { return _mm_sub_epi64(a, b); }
#endif

#ifdef DPM_HAS_AVX
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 add(__m256 a, __m256 b) noexcept { return _mm256_add_ps(a, b); }
	template<std::same_as<float> T>
	[[nodiscard]] DPM_FORCEINLINE __m256 sub(__m256 a, __m256 b) noexcept { return _mm256_sub_ps(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d add(__m256d a, __m256d b) noexcept { return _mm256_add_pd(a, b); }
	template<std::same_as<double> T>
	[[nodiscard]] DPM_FORCEINLINE __m256d sub(__m256d a, __m256d b) noexcept { return _mm256_sub_pd(a, b); }

#ifdef DPM_HAS_AVX2
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return _mm256_add_epi8(a, b); }
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return _mm256_sub_epi8(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return _mm256_add_epi16(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return _mm256_sub_epi16(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return _mm256_add_epi32(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return _mm256_sub_epi32(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return _mm256_add_epi64(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return _mm256_sub_epi64(a, b); }
#else
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return add<T>(a, b); } , a, b); }
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return sub<T>(a, b); }, a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return add<T>(a, b); }, a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return sub<T>(a, b); }, a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return add<T>(a, b); }, a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return sub<T>(a, b); }, a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i add(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return add<T>(a, b); }, a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i sub(__m256i a, __m256i b) noexcept { return mux_128x2<__m256i>([](auto a, auto b){return sub<T>(a, b); }, a, b); }
#endif
#endif

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V inc(V x) noexcept { return add<T>(x, fill<V>(T{1})); }
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V dec(V x) noexcept { return sub<T>(x, fill<V>(T{1})); }
	template<std::floating_point T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V negate(V x) noexcept { return bit_xor(x, fill<V>(-0.0)); }
	template<std::integral T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V negate(V x) noexcept { return sub<T>(setzero<V>(), x); }
}