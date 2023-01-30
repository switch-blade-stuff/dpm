/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "type_fwd.hpp"

#ifdef DPM_HAS_SSE2

namespace dpm::detail
{
	[[nodiscard]] DPM_FORCEINLINE __m128 cvt_u32_f32(__m128i x) noexcept
	{
		const auto a = _mm_cvtepi32_ps(_mm_and_si128(x, _mm_set1_epi32(1)));
		const auto b = _mm_cvtepi32_ps(_mm_srli_epi32(x, 1));
		return _mm_add_ps(_mm_add_ps(b, b), a); /* (x >> 1) * 2 + x & 1 */
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i cvt_f32_u32(__m128 x) noexcept
	{
		const auto offset = _mm_set1_ps(0x40'0000);
		return std::bit_cast<__m128i>(_mm_xor_ps(_mm_add_ps(x, offset), offset));
	}

	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d cvt_u64_f64_sse(__m128i x) noexcept
	{
		const auto exp84 = std::bit_cast<__m128i>(_mm_set1_pd(19342813113834066795298816.));  /* 2^84 */
		const auto exp52 = std::bit_cast<__m128i>(_mm_set1_pd(0x0010'0000'0000'0000));        /* 2^52 */
		const auto adjust = _mm_set1_pd(19342813118337666422669312.);                         /* 2^84 + 2^52 */

		const auto a = _mm_or_si128(_mm_srli_epi64(x, 32), exp84);
		const auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000));
		const auto b = _mm_or_si128(_mm_and_si128(mask, exp52), _mm_andnot_si128(mask, x));

		return _mm_add_pd(_mm_sub_pd(std::bit_cast<__m128d>(a), adjust), std::bit_cast<__m128d>(b));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d cvt_i64_f64_sse(__m128i x) noexcept
	{
		const auto exp67m3 = std::bit_cast<__m128i>(_mm_set1_pd(442721857769029238784.)); /* 2^67 * 3 */
		const auto exp52 = std::bit_cast<__m128i>(_mm_set1_pd(0x0010'0000'0000'0000));    /* 2^52 */
		const auto adjust = _mm_set1_pd(442726361368656609280.);                          /* 2^67 * 3 + 2^52 */

		const auto a = _mm_and_si128(_mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000)), _mm_srai_epi32(x, 16));
		const auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'0000'0000'0000));
		const auto b = _mm_or_si128(_mm_and_si128(mask, exp52), _mm_andnot_si128(mask, x));

		return _mm_add_pd(_mm_sub_pd(std::bit_cast<__m128d>(_mm_add_epi64(a, exp67m3)), adjust), std::bit_cast<__m128d>(b));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_u64_sse(__m128d x) noexcept
	{
		const auto offset = _mm_set1_pd(0x0010'0000'0000'0000);
		return std::bit_cast<__m128i>(_mm_xor_pd(_mm_add_pd(x, offset), offset));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_i64_sse(__m128d x) noexcept
	{
		const auto offset = _mm_set1_pd(0x0018'0000'0000'0000);
		return _mm_sub_epi64(std::bit_cast<__m128i>(_mm_add_pd(x, offset)), std::bit_cast<__m128i>(offset));
	}

	[[nodiscard]] DPM_FORCEINLINE __m128d cvt_u64_f64(__m128i x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtepu64_pd(x);
#else
		return cvt_u64_f64_sse(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m128d cvt_i64_f64(__m128i x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtepi64_pd(x);
#else
		return cvt_i64_f64_sse(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_u64(__m128d x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtpd_epu64(x);
#else
		return cvt_f64_u64_sse(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m128i cvt_f64_i64(__m128d x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtpd_epi64(x);
#else
		return cvt_f64_i64_sse(x);
#endif
	}

#ifdef DPM_HAS_AVX
	[[nodiscard]] DPM_FORCEINLINE __m256i cvt_f32_u32(__m256 x) noexcept
	{
		const auto offset = _mm256_set1_ps(0x40'0000);
		return std::bit_cast<__m256i>(_mm256_xor_ps(_mm256_add_ps(x, offset), offset));
	}
	[[nodiscard]] DPM_FORCEINLINE __m256 cvt_u32_f32(__m256i x) noexcept
	{
#ifdef DPM_HAS_AVX2
		const auto a = _mm256_cvtepi32_ps(_mm256_and_si256(x, _mm256_set1_epi32(1)));
		const auto b = _mm256_cvtepi32_ps(_mm256_srli_epi32(x, 1));
		return _mm256_add_ps(_mm256_add_ps(b, b), a); /* (x >> 1) * 2 + x & 1 */
#else
		const auto l = cvt_f32_u32(_mm256_extractf128_si256(x, 0));
		const auto h = cvt_f32_u32(_mm256_extractf128_si256(x, 1));
		return _mm256_set_m128(h, l);
#endif
	}

	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256i cvt_f64_u64_avx(__m256d x) noexcept
	{
		const auto offset = _mm256_set1_pd(0x0010'0000'0000'0000);
		return std::bit_cast<__m256i>(_mm256_xor_pd(_mm256_add_pd(x, offset), offset));
	}

#ifdef DPM_HAS_AVX2
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d cvt_u64_f64_avx(__m256i x) noexcept
	{
		const auto exp84 = std::bit_cast<__m256i>(_mm256_set1_pd(19342813113834066795298816.));  /* 2^84 */
		const auto exp52 = std::bit_cast<__m256i>(_mm256_set1_pd(0x0010'0000'0000'0000));        /* 2^52 */
		const auto adjust = _mm256_set1_pd(19342813118337666422669312.);                         /* 2^84 + 2^52 */

		const auto a = _mm256_or_si256(_mm256_srli_epi64(x, 32), exp84);
		const auto mask = _mm256_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000));
		const auto b = _mm256_or_si256(_mm256_and_si256(mask, exp52), _mm256_andnot_si256(mask, x));

		return _mm256_add_pd(_mm256_sub_pd(std::bit_cast<__m256d>(a), adjust), std::bit_cast<__m256d>(b));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d cvt_i64_f64_avx(__m256i x) noexcept
	{
		const auto exp67m3 = std::bit_cast<__m256i>(_mm256_set1_pd(442721857769029238784.)); /* 2^67 * 3 */
		const auto exp52 = std::bit_cast<__m256i>(_mm256_set1_pd(0x0010'0000'0000'0000));    /* 2^52 */
		const auto adjust = _mm256_set1_pd(442726361368656609280.);                          /* 2^67 * 3 + 2^52 */

		const auto a = _mm256_and_si256(_mm256_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'0000'0000)), _mm256_srai_epi32(x, 16));
		const auto mask = _mm256_set1_epi64x(static_cast<std::int64_t>(0xffff'0000'0000'0000));
		const auto b = _mm256_or_si256(_mm256_and_si256(mask, exp52), _mm256_andnot_si256(mask, x));

		return _mm256_add_pd(_mm256_sub_pd(std::bit_cast<__m256d>(_mm256_add_epi64(a, exp67m3)), adjust), std::bit_cast<__m256d>(b));
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256i cvt_f64_i64_avx(__m256d x) noexcept
	{
		const auto offset = _mm256_set1_pd(0x0018'0000'0000'0000);
		return _mm256_sub_epi64(std::bit_cast<__m256i>(_mm256_add_pd(x, offset)), std::bit_cast<__m256i>(offset));
	}
#else
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d cvt_u64_f64_avx(__m256i x) noexcept
	{
		const auto l = cvt_u64_f64_sse(_mm256_extractf128_si256(x, 0));
		const auto h = cvt_u64_f64_sse(_mm256_extractf128_si256(x, 1));
		return _mm256_set_m128d(h, l);
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d cvt_i64_f64_avx(__m256i x) noexcept
	{
		const auto l = cvt_i64_f64_sse(_mm256_extractf128_si256(x, 0));
		const auto h = cvt_i64_f64_sse(_mm256_extractf128_si256(x, 1));
		return _mm256_set_m128d(h, l);
	}
	[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256i cvt_f64_i64_avx(__m256d x) noexcept
	{
		const auto l = cvt_f64_i64_sse(_mm256_extractf128_pd(x, 0));
		const auto h = cvt_f64_i64_sse(_mm256_extractf128_pd(x, 1));
		return _mm256_set_m128i(h, l);
	}
#endif

	[[nodiscard]] DPM_FORCEINLINE __m256d cvt_u64_f64(__m256i x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm256_cvtepu64_pd(x);
#else
		return cvt_u64_f64_avx(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m256d cvt_i64_f64(__m256i x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtepi64_pd(x);
#else
		return cvt_i64_f64_avx(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m256i cvt_f64_u64(__m256d x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtpd_epu64(x);
#else
		return cvt_f64_u64_avx(x);
#endif
	}
	[[nodiscard]] DPM_FORCEINLINE __m256i cvt_f64_i64(__m256d x) noexcept
	{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
		return _mm_cvtpd_epi64(x);
#else
		return cvt_f64_i64_avx(x);
#endif
	}
#endif
}

#endif