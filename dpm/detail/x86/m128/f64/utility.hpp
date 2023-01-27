/*
 * Created by switchblade on 2023-01-12.
 */

#pragma once

#include "../../type_fwd.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm::detail
{
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d cvt_u64_f64_sse2(__m128i x) noexcept
	{
		const auto exp84 = std::bit_cast<__m128i>(_mm_set1_pd(19342813113834066795298816.));  /* 2^84 */
		const auto exp52 = std::bit_cast<__m128i>(_mm_set1_pd(0x0010'0000'0000'0000));        /* 2^52 */
		const auto adjust = _mm_set1_pd(19342813118337666422669312.);                         /* 2^84 + 2^52 */

		const auto a = _mm_or_si128(_mm_srli_epi64(x, 32), exp84);
		const auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000));
		const auto b = _mm_or_si128(_mm_and_si128(mask, exp52), _mm_andnot_si128(mask, x));

		return _mm_add_pd(_mm_sub_pd(std::bit_cast<__m128d>(a), adjust), std::bit_cast<__m128d>(b));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d cvt_i64_f64_sse2(__m128i x) noexcept
	{
		const auto exp67m3 = std::bit_cast<__m128i>(_mm_set1_pd(442721857769029238784.)); /* 2^67 * 3 */
		const auto exp52 = std::bit_cast<__m128i>(_mm_set1_pd(0x0010'0000'0000'0000));    /* 2^52 */
		const auto adjust = _mm_set1_pd(442726361368656609280.);                          /* 2^67 * 3 + 2^52 */

		const auto a = _mm_and_si128(_mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000)), _mm_srai_epi32(x, 16));
		const auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'0000'0000'0000));
		const auto b = _mm_or_si128(_mm_and_si128(mask, exp52), _mm_andnot_si128(mask, x));

		return _mm_add_pd(_mm_sub_pd(std::bit_cast<__m128d>(_mm_add_epi64(a, exp67m3)), adjust), std::bit_cast<__m128d>(b));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_u64_sse2(__m128d x) noexcept
	{
		const auto offset = _mm_set1_pd(0x0010'0000'0000'0000);
		return _mm_xor_si128(std::bit_cast<__m128i>(_mm_add_pd(x, offset)), std::bit_cast<__m128i>(offset));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_i64_sse2(__m128d x) noexcept
	{
		const auto offset = _mm_set1_pd(0x0018'0000'0000'0000);
		return _mm_sub_epi64(std::bit_cast<__m128i>(_mm_add_pd(x, offset)), std::bit_cast<__m128i>(offset));
	}

	[[nodiscard]] DPM_FORCEINLINE __m128d cvt_u64_f64(__m128i x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ)
		return _mm_cvtepu64_pd(x);
#else
		return cvt_u64_f64_sse2(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m128d cvt_i64_f64(__m128i x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ)
		return _mm_cvtepi64_pd(x);
#else
		return cvt_i64_f64_sse2(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_u64(__m128d x) noexcept
	{
#ifdef DPM_HAS_AVX512DQ
		return _mm_cvtpd_epu64(x);
#else
		return cvt_f64_u64_sse2(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_i64(__m128d x) noexcept
	{
#ifdef DPM_HAS_AVX512DQ
		return _mm_cvtpd_epi64(x);
#else
		return cvt_f64_i64_sse2(x);
#endif
	}
}

#endif