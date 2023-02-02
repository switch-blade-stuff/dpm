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

	template<integral_of_size<2> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i x) noexcept { return _mm_slli_epi16(x, N); }
	template<integral_of_size<2> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i x) noexcept { return _mm_srli_epi16(x, N); }
	template<integral_of_size<4> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i x) noexcept { return _mm_slli_epi32(x, N); }
	template<integral_of_size<4> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i x) noexcept { return _mm_srli_epi32(x, N); }
	template<integral_of_size<8> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i x) noexcept { return _mm_slli_epi64(x, N); }
	template<integral_of_size<8> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i x) noexcept { return _mm_srli_epi64(x, N); }

	/* Emulate AVX2 32-bit shifts via scalar shifts. These are only used internally. */
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl32_sse(__m128i a, __m128i b) noexcept
	{
		auto b_data = reinterpret_cast<const alias_t<std::uint32_t> *>(&b);
		auto a_data = reinterpret_cast<alias_t<std::uint32_t> *>(&a);
		a_data[0] <<= b_data[0];
		a_data[1] <<= b_data[1];
		a_data[2] <<= b_data[2];
		a_data[3] <<= b_data[3];
		return a;
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr32_sse(__m128i a, __m128i b) noexcept
	{
		auto b_data = reinterpret_cast<const alias_t<std::uint32_t> *>(&b);
		auto a_data = reinterpret_cast<alias_t<std::uint32_t> *>(&a);
		a_data[0] >>= b_data[0];
		a_data[1] >>= b_data[1];
		a_data[2] >>= b_data[2];
		a_data[3] >>= b_data[3];
		return a;
	}

	/* Emulate AVX2 64-bit shifts by shifting individual elements. */
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl64_sse(__m128i a, __m128i b) noexcept
	{
		const auto sl = _mm_sll_epi64(a, b);
		const auto bh = _mm_unpackhi_epi64(b, b);
		const auto sh = _mm_sll_epi64(a, bh);
		return _mm_shuffle_pd(sl, sh, 0b10);
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr64_sse(__m128i a, __m128i b) noexcept
	{
		const auto sl = _mm_srl_epi64(a, b);
		const auto bh = _mm_unpackhi_epi64(b, b);
		const auto sh = _mm_srl_epi64(a, bh);
		return _mm_shuffle_pd(sl, sh, 0b10);
	}

#ifndef DPM_HAS_AVX2
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i a, __m128i b) noexcept { return bit_shiftl64_sse(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i a, __m128i b) noexcept { return bit_shiftr64_sse(a, b); }
#endif
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

	template<integral_of_size<2> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i x) noexcept { return _mm256_slli_epi16(x, N); }
	template<integral_of_size<2> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i x) noexcept { return _mm256_srli_epi16(x, N); }
	template<integral_of_size<4> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i x) noexcept { return _mm256_slli_epi32(x, N); }
	template<integral_of_size<4> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i x) noexcept { return _mm256_srli_epi32(x, N); }
	template<integral_of_size<8> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i x) noexcept { return _mm256_slli_epi64(x, N); }
	template<integral_of_size<8> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i x) noexcept { return _mm256_srli_epi64(x, N); }

	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i a, __m128i b) noexcept { return _mm_sllv_epi32(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i a, __m128i b) noexcept { return _mm_srlv_epi32(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i a, __m256i b) noexcept { return _mm256_sllv_epi32(a, b); }
	template<integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i a, __m256i b) noexcept { return _mm256_srlv_epi32(a, b); }

	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i a, __m128i b) noexcept { return _mm_sllv_epi64(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i a, __m128i b) noexcept { return _mm_srlv_epi64(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i a, __m256i b) noexcept { return _mm256_sllv_epi64(a, b); }
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i a, __m256i b) noexcept { return _mm256_srlv_epi64(a, b); }
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

	template<integral_of_size<2> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i x) noexcept { return mux_128x2<__m256i>([](auto x) { return bit_shiftl<T, N>(x); }, x); }
	template<integral_of_size<2> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i x) noexcept { return mux_128x2<__m256i>([](auto x) { return bit_shiftr<T, N>(x); }, x); }
	template<integral_of_size<4> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i x) noexcept { return mux_128x2<__m256i>([](auto x) { return bit_shiftl<T, N>(x); }, x); }
	template<integral_of_size<4> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i x) noexcept { return mux_128x2<__m256i>([](auto x) { return bit_shiftr<T, N>(x); }, x); }
	template<integral_of_size<8> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i x) noexcept { return mux_128x2<__m256i>([](auto x) { return bit_shiftl<T, N>(x); }, x); }
	template<integral_of_size<8> T, int N>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i x) noexcept { return mux_128x2<__m256i>([](auto x) { return bit_shiftr<T, N>(x); }, x); }

	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i a, __m256i b) noexcept
	{
		return mux_128x2<__m256i>([](auto a, auto b) { return bit_shiftl64_sse(a, b); }, a, b);
	}
	template<integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i a, __m256i b) noexcept
	{
		return mux_128x2<__m256i>([](auto a, auto b) { return bit_shiftr64_sse(a, b); }, a, b);
	}
#endif
#endif

#if defined(DPM_HAS_AVX512BW) && defined(DPM_HAS_AVX512VL)
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftl(__m128i a, __m128i b) noexcept { return _mm_sllv_epi16(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i bit_shiftr(__m128i a, __m128i b) noexcept { return _mm_srlv_epi16(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftl(__m256i a, __m256i b) noexcept { return _mm256_sllv_epi16(a, b); }
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i bit_shiftr(__m256i a, __m256i b) noexcept { return _mm256_srlv_epi16(a, b); }
#endif

	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V bit_not(V x) noexcept { return bit_xor(x, setones<V>()); }
}