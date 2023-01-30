/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "utility.hpp"

namespace dpm::detail
{
	[[nodiscard]] DPM_FORCEINLINE __m128 bit_and(__m128 a, __m128 b) noexcept { return _mm_and_ps(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m128 bit_xor(__m128 a, __m128 b) noexcept { return _mm_xor_ps(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m128 bit_or(__m128 a, __m128 b) noexcept { return _mm_or_ps(a, b); }

#ifdef DPM_HAS_SSE2
	[[nodiscard]] DPM_FORCEINLINE __m128d bit_and(__m128d a, __m128d b) noexcept { return _mm_and_pd(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m128d bit_xor(__m128d a, __m128d b) noexcept { return _mm_xor_pd(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m128d bit_or(__m128d a, __m128d b) noexcept { return _mm_or_pd(a, b); }

	[[nodiscard]] DPM_FORCEINLINE __m128i bit_and(__m128i a, __m128i b) noexcept { return _mm_and_si128(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_xor(__m128i a, __m128i b) noexcept { return _mm_xor_si128(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_or(__m128i a, __m128i b) noexcept { return _mm_or_si128(a, b); }
#endif

#ifdef DPM_HAS_AVX
	[[nodiscard]] DPM_FORCEINLINE __m256 bit_and(__m256 a, __m256 b) noexcept { return _mm256_and_ps(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m256 bit_xor(__m256 a, __m256 b) noexcept { return _mm256_xor_ps(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m256 bit_or(__m256 a, __m256 b) noexcept { return _mm256_or_ps(a, b); }

	[[nodiscard]] DPM_FORCEINLINE __m256d bit_and(__m256d a, __m256d b) noexcept { return _mm256_and_pd(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m256d bit_xor(__m256d a, __m256d b) noexcept { return _mm256_xor_pd(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m256d bit_or(__m256d a, __m256d b) noexcept { return _mm256_or_pd(a, b); }

#ifdef DPM_HAS_AVX2
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_and(__m256i a, __m256i b) noexcept { return _mm256_and_si256(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_xor(__m256i a, __m256i b) noexcept { return _mm256_xor_si256(a, b); }
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_or(__m256i a, __m256i b) noexcept { return _mm256_or_si256(a, b); }
#else
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_and(__m256i a, __m256i b)
	{
		const auto af = std::bit_cast<__m256>(a);
		const auto bf = std::bit_cast<__m256>(b);
		return std::bit_cast<__m256i>(bit_and(af, bf));
	}
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_xor(__m256i a, __m256i b)
	{
		const auto af = std::bit_cast<__m256>(a);
		const auto bf = std::bit_cast<__m256>(b);
		return std::bit_cast<__m256i>(bit_xor(af, bf));
	}
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_or(__m256i a, __m256i b)
	{
		const auto af = std::bit_cast<__m256>(a);
		const auto bf = std::bit_cast<__m256>(b);
		return std::bit_cast<__m256i>(bit_or(af, bf));
	}
#endif
#endif

	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V bit_not(V x) noexcept { return bit_xor(x, setones<V>()); }
}