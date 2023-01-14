/*
 * Created by switchblade on 2023-01-12.
 */

#pragma once

#include "../../../define.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include <smmintrin.h>

#ifndef DPM_USE_IMPORT

#include <bit>

#endif

namespace dpm::detail
{
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_floor_f64(__m128d x) noexcept
	{
#ifdef DPM_HAS_SSE4_1
		return _mm_floor_pd(x);
#else
		const auto exp52 = _mm_set1_pd(0x0010'0000'0000'0000);
		const auto mask = _mm_cmpnlt_pd(x, exp52);

		const auto magic = _mm_set1_pd(std::bit_cast<double>(0x4338'0000'0000'0000));
		const auto a = _mm_sub_pd(_mm_add_pd(x, magic), magic);
		const auto b = _mm_and_pd(_mm_cmplt_pd(x, a), _mm_set1_pd(1.0));
		const auto result = _mm_sub_pd(a, b);

		return _mm_or_pd(_mm_and_pd(mask, x), _mm_andnot_pd(mask, result));
#endif
	}

	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_cvt_u64_f64(__m128i x) noexcept
	{
#ifdef DPM_HAS_AVX512DQ
		return _mm_cvtepu64_pd(x);
#else
		const auto exp84 = std::bit_cast<__m128i>(_mm_set1_pd(19342813113834066795298816.));  /* 2^84 */
		const auto exp52 = std::bit_cast<__m128i>(_mm_set1_pd(0x0010'0000'0000'0000));        /* 2^52 */
		const auto adjust = _mm_set1_pd(19342813118337666422669312.);                         /* 2^84 + 2^52 */

		const auto a = _mm_or_si128(_mm_srli_epi64(x, 32), exp84);
#ifdef DPM_HAS_SSE4_1
		const auto b = _mm_blend_epi16(x, exp52, 0xcc);
#else
		const auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000));
		const auto b = _mm_or_si128(_mm_and_si128(mask, exp52), _mm_andnot_si128(mask, x));
#endif
		return _mm_add_pd(_mm_sub_pd(std::bit_cast<__m128d>(a), adjust), std::bit_cast<__m128d>(b));
#endif
	}
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_cvt_i64_f64(__m128i x) noexcept
	{
#ifdef DPM_HAS_AVX512DQ
		return _mm_cvtepi64_pd(x);
#else
		const auto exp67m3 = std::bit_cast<__m128i>(_mm_set1_pd(442721857769029238784.)); /* 2^67 * 3 */
		const auto exp52 = std::bit_cast<__m128i>(_mm_set1_pd(0x0010'0000'0000'0000));    /* 2^52 */
		const auto adjust = _mm_set1_pd(442726361368656609280.);                          /* 2^67 * 3 + 2^52 */

		const auto temp = _mm_srai_epi32(x, 16);
#ifdef DPM_HAS_SSE4_1
		auto a = _mm_blend_epi16(temp, _mm_setzero_si128(), 0x33);
#else
		auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0x0000'0000'ffff'ffff));
		auto a = _mm_or_si128(_mm_and_si128(mask, _mm_setzero_si128()), _mm_andnot_si128(mask, temp));
#endif

#ifdef DPM_HAS_SSE4_1
		auto b = _mm_blend_epi16(x, exp52, 0x88);
#else
		mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'0000'0000'0000));
		const auto b = _mm_or_si128(_mm_and_si128(mask, exp52), _mm_andnot_si128(mask, x));
#endif

		return _mm_add_pd(_mm_sub_pd(std::bit_cast<__m128d>(_mm_add_epi64(a, exp67m3)), adjust), std::bit_cast<__m128d>(b));
#endif
	}
	[[nodiscard]] inline __m128i DPM_FORCEINLINE x86_cvt_f64_u64(__m128d x) noexcept
	{
#ifdef DPM_HAS_AVX512DQ
		return _mm_cvtpd_epu64(x);
#else
		const auto offset = _mm_set1_pd(0x0010'0000'0000'0000);
		return _mm_xor_si128(std::bit_cast<__m128i>(_mm_add_pd(x, offset)), std::bit_cast<__m128i>(offset));
#endif
	}
	[[nodiscard]] inline __m128i DPM_FORCEINLINE x86_cvt_f64_i64(__m128d x) noexcept
	{
#ifdef DPM_HAS_AVX512DQ
		return _mm_cvtpd_epi64(x);
#else
		const auto offset = _mm_set1_pd(0x0018'0000'0000'0000);
		return _mm_sub_epi64(std::bit_cast<__m128i>(_mm_add_pd(x, offset)), std::bit_cast<__m128i>(offset));
#endif
	}
}

#endif