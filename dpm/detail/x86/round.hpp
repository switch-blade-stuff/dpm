/*
 * Created by switchblade on 2023-02-07.
 */

#pragma once

#include "math_fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "bitwise.hpp"
#include "class.hpp"
#include "cvt.hpp"

namespace dpm
{
	namespace detail
	{
#ifndef DPM_HAS_SSE4_1
		[[nodiscard]] DPM_FORCEINLINE __m128 mask_domain(__m128 x, __m128 result) noexcept
		{
			const auto fin_mask = isfinite(x);
			return _mm_or_ps(_mm_andnot_ps(fin_mask, x), _mm_and_ps(fin_mask, result));
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d mask_domain(__m128d x, __m128d result) noexcept
		{
			const auto fin_mask = isfinite(x);
			return _mm_or_pd(_mm_andnot_pd(fin_mask, x), _mm_and_pd(fin_mask, result));
		}
#endif

		[[nodiscard]] DPM_FORCEINLINE __m128 ceil(__m128 x) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			return _mm_ceil_ps(x);
#elif defined(DPM_USE_SVML)
			return _mm_svml_ceil_ps(x);
#else
			const auto tx = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
			return mask_domain(x, _mm_add_ps(tx, bit_and(_mm_cmplt_ps(tx, x), _mm_set1_ps(1.0f))));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 floor(__m128 x) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			return _mm_floor_ps(x);
#elif defined(DPM_USE_SVML)
			return _mm_svml_floor_ps(x);
#else
			const auto tx = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
			return mask_domain(x, _mm_sub_ps(tx, bit_and(_mm_cmpgt_ps(tx, x), _mm_set1_ps(1.0f))));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 trunc(__m128 x) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			return _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#elif defined(DPM_USE_SVML)
			return _mm_trunc_ps(x);
#else
			return mask_domain(x, _mm_cvtepi32_ps(_mm_cvttps_epi32(x)));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 nearbyint(__m128 x) noexcept
		{
#ifdef DPM_HAS_SSE4_1
			return _mm_round_ps(x, _MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION);
#else
			return mask_domain(x, _mm_cvtepi32_ps(_mm_cvtps_epi32(x)));
#endif
		}

		[[nodiscard]] DPM_FORCEINLINE __m128d ceil(__m128d x) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			return _mm_ceil_pd(x);
#elif defined(DPM_USE_SVML)
			return _mm_svml_ceil_pd(x);
#else
			const auto tx = _mm_cvtepi32_pd(_mm_cvttpd_epi32(x));
			return mask_domain(x, _mm_add_pd(tx, bit_and(_mm_cmplt_pd(tx, x), _mm_set1_pd(1.0f))));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d floor(__m128d x) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			return _mm_floor_pd(x);
#elif defined(DPM_USE_SVML)
			return _mm_svml_floor_pd(x);
#else
			const auto tx = _mm_cvtepi32_pd(_mm_cvttpd_epi32(x));
			return mask_domain(x, _mm_sub_pd(tx, bit_and(_mm_cmpgt_pd(tx, x), _mm_set1_pd(1.0f))));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d trunc(__m128d x) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			return _mm_round_ps(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#elif defined(DPM_USE_SVML)
			return _mm_trunc_pd(x);
#else
			return mask_domain(x, _mm_cvtepi32_pd(_mm_cvttpd_epi32(x)));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d nearbyint(__m128d x) noexcept
		{
#ifdef DPM_HAS_SSE4_1
			return _mm_round_ps(x, _MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION);
#else
			return mask_domain(x, _mm_cvtepi32_pd(_mm_cvtpd_epi32(x)));
#endif
		}

		template<integral_of_size<4> T>
		DPM_FORCEINLINE void round(__m128 x, __m128i *dst) noexcept
		{
			const auto ix = _mm_cvttps_epi32(x);
			const auto tx = _mm_cvtepi32_ps(ix);
			const auto near2 = _mm_set1_ps(std::bit_cast<float>(0x3fffffff));
			const auto rem = _mm_mul_ps(_mm_sub_ps(x, tx), near2);
			*dst = _mm_add_epi32(ix, _mm_cvttps_epi32(rem));
		}
		template<integral_of_size<8> T>
		DPM_FORCEINLINE void round(__m128 x, __m128i *dst) noexcept
		{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
			auto ih = _mm_cvttps_epi64(_mm_movehl_ps(x, x));
			auto il = _mm_cvttps_epi64(x);
			const auto th = _mm_cvtepi64_ps(ih);
			const auto tl = _mm_cvtepi64_ps(il);
			const auto tx = _mm_movelh_ps(tl, th);

			const auto near2 = _mm_set1_ps(std::bit_cast<float>(0x3fffffff));
			const auto rem = _mm_mul_ps(_mm_sub_ps(x, tx), near2);
			dst[1] = _mm_add_epi64(th, _mm_cvttps_epi64(_mm_movehl_ps(rem, rem)));
			dst[0] = _mm_add_epi64(tl, _mm_cvttps_epi64(rem));
#else
			__m128i tmp;
			round<std::int32_t>(x, &tmp);
			dst[0] = std::bit_cast<__m128i>(_mm_unpacklo_ps(std::bit_cast<__m128>(tmp), _mm_setzero_ps()));
			dst[1] = std::bit_cast<__m128i>(_mm_unpackhi_ps(std::bit_cast<__m128>(tmp), _mm_setzero_ps()));
#endif
		}
		template<std::same_as<float> T>
		DPM_FORCEINLINE void round(__m128 x, __m128 *dst) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			const auto sign = bit_and(x, _mm_set1_ps(-0.0f));
			const auto off = bit_or(sign, std::bit_cast<__m128>(_mm_set1_epi32(0x3effffff)));
			*dst = _mm_round_ps(_mm_add_ps(x, off), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#elif defined(DPM_USE_SVML)
			*dst = _mm_svml_round_ps(x);
#else
			const auto tx = _mm_cvtepi32_ps(_mm_cvttps_epi32(x));
			const auto near2 = _mm_set1_ps(std::bit_cast<float>(0x3fffffff));
			const auto rem = _mm_mul_ps(_mm_sub_ps(x, tx), near2);
			*dst = mask_domain(x, _mm_add_ps(tx, _mm_cvtepi32_ps(_mm_cvttps_epi32(rem))));
#endif
		}

		template<integral_of_size<4> T>
		DPM_FORCEINLINE void rint(__m128 x, __m128i *dst) noexcept { *dst = _mm_cvtps_epi32(x); }
		template<integral_of_size<8> T>
		DPM_FORCEINLINE void rint(__m128 x, __m128i *dst) noexcept
		{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
			dst[1] = _mm_cvtps_epi64(_mm_movehl_ps(x, x));
			dst[0] = _mm_cvtps_epi64(x);
#else
			const auto tmp = _mm_cvtps_epi32(x);
			dst[0] = std::bit_cast<__m128i>(_mm_unpacklo_ps(tmp, _mm_setzero_ps()));
			dst[1] = std::bit_cast<__m128i>(_mm_unpackhi_ps(tmp, _mm_setzero_ps()));
#endif
		}
		template<std::same_as<float> T>
		DPM_FORCEINLINE void rint(__m128 x, __m128 *dst) noexcept
		{
#ifdef DPM_HAS_SSE4_1
			*dst = _mm_round_ps(x, _MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION);
#else
			const auto sign_x = _mm_and_ps(x, _mm_set1_ps(-0.0));
			const auto x_offset = _mm_set1_ps(8388608.0);
			const auto abs_x = _mm_xor_ps(x, sign_x);

			const auto tmp_mask = _mm_cmpge_ps(abs_x, x_offset);
			const auto tmp = _mm_or_ps(_mm_sub_ps(_mm_add_ps(abs_x, x_offset), x_offset), sign_x);
			*dst = _mm_or_ps(_mm_andnot_ps(tmp_mask, tmp), _mm_and_ps(tmp_mask, x));
#endif
		}

		template<integral_of_size<4> T>
		DPM_FORCEINLINE void round(__m128d x, __m128i *dst) noexcept
		{
			const auto ix = _mm_cvttpd_epi32(x);
			const auto tx = _mm_cvtepi32_pd(ix);
			const auto near2 = _mm_set1_pd(std::bit_cast<double>(0x3fffffff'ffffffff));
			const auto rem = _mm_mul_pd(_mm_sub_pd(x, tx), near2);
			*dst = _mm_add_epi32(ix, _mm_cvttpd_epi32(rem));
		}
		template<integral_of_size<8> T>
		DPM_FORCEINLINE void round(__m128d x, __m128i *dst) noexcept
		{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
			const auto ix = _mm_cvttpd_epi64(x);
			const auto tx = _mm_cvtepi64_pd(ix);
			const auto near2 = _mm_set1_pd(std::bit_cast<double>(0x3fffffff'ffffffff));
			const auto rem = _mm_mul_pd(_mm_sub_pd(x, tx), near2);
			*dst = _mm_add_epi64(ix, _mm_cvttpd_epi64(rem));
#else
			__m128i tmp;
			round<std::int32_t>(x, &tmp);
			dst[0] = std::bit_cast<__m128i>(_mm_unpacklo_ps(std::bit_cast<__m128>(tmp), _mm_setzero_ps()));
			dst[1] = std::bit_cast<__m128i>(_mm_unpackhi_ps(std::bit_cast<__m128>(tmp), _mm_setzero_ps()));
#endif
		}
		template<std::same_as<double> T>
		DPM_FORCEINLINE void round(__m128d x, __m128d *dst) noexcept
		{
#if defined(DPM_HAS_SSE4_1)
			const auto sign = bit_and(x, _mm_set1_pd(-0.0f));
			const auto off = bit_or(sign, std::bit_cast<__m128d>(_mm_set1_epi64x(0x3fdfffff'ffffffff)));
			*dst = _mm_round_ps(_mm_add_pd(x, off), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#elif defined(DPM_USE_SVML)
			*dst = _mm_svml_round_pd(x);
#else
			const auto tx = _mm_cvtepi32_pd(_mm_cvttpd_epi32(x));
			const auto near2 = _mm_set1_pd(std::bit_cast<double>(0x3fffffff'ffffffff));
			const auto rem = _mm_mul_pd(_mm_sub_pd(x, tx), near2);
			*dst = mask_domain(x, _mm_add_pd(tx, _mm_cvtepi32_pd(_mm_cvttpd_epi32(rem))));
#endif
		}

		template<integral_of_size<4> T>
		DPM_FORCEINLINE void rint(__m128d x, __m128i *dst) noexcept { *dst = _mm_cvtpd_epi32(x); }
		template<integral_of_size<8> T>
		DPM_FORCEINLINE void rint(__m128d x, __m128i *dst) noexcept { *dst = cvt_f64_i64(x); }
		template<std::same_as<double> T>
		DPM_FORCEINLINE void rint(__m128d x, __m128d *dst) noexcept
		{
#ifdef DPM_HAS_SSE4_1
			*dst = _mm_round_ps(x, _MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION);
#else
			const auto sign_x = _mm_and_pd(x, _mm_set1_pd(-0.0));
			const auto x_offset = _mm_set1_pd(4503599627370496.0);
			const auto abs_x = _mm_xor_pd(x, sign_x);

			const auto tmp_mask = _mm_cmpge_pd(abs_x, x_offset);
			const auto tmp = _mm_or_pd(_mm_sub_pd(_mm_add_pd(abs_x, x_offset), x_offset), sign_x);
			*dst = _mm_or_pd(_mm_andnot_pd(tmp_mask, tmp), _mm_and_pd(tmp_mask, tmp));
#endif
		}

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 ceil(__m256 x) noexcept { return _mm256_ceil_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 floor(__m256 x) noexcept { return _mm256_floor_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 round(__m256 x) noexcept
		{
			const auto sign = bit_and(x, _mm256_set1_ps(-0.0f));
			const auto off = bit_or(sign, std::bit_cast<__m256>(_mm256_set1_epi32(0x3effffff)));
			return _mm256_round_ps(bit_and(x, off), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 nearbyint(__m256 x) noexcept { return _mm256_round_ps(x, _MM_FROUND_NO_EXC | _MM_FROUND_CUR_DIRECTION); }

		template<integral_of_size<4> T>
		DPM_FORCEINLINE void round(__m256 x, __m256i *dst) noexcept
		{
			const auto ix = _mm256_cvttps_epi32(x);
			const auto tx = _mm256_cvtepi32_ps(ix);
			const auto near2 = _mm256_set1_ps(std::bit_cast<float>(0x3fffffff));
			const auto rem = _mm256_mul_ps(_mm256_sub_ps(x, tx), near2);
#ifdef DPM_HAS_AVX2
			*dst = _mm256_add_epi32(ix, _mm256_cvttps_epi32(rem));
#else
			*dst = mux_128x2<__m256i>([](auto i) { return _mm_add_epi32(i); }, _mm256_cvttps_epi32(rem));
#endif
		}
		template<integral_of_size<8> T>
		DPM_FORCEINLINE void round(__m256 x, __m256i *dst) noexcept
		{
#if defined(DPM_HAS_AVX512DQ) && defined(DPM_HAS_AVX512VL)
			const auto ih = _mm256_cvttps_epi64(_mm256_permute2f128_ps(x, x, 0x11));
			const auto il = _mm256_cvttps_epi64(x);
			const auto th = _mm256_cvtepi64_ps(ih);
			const auto tl = _mm256_cvtepi64_ps(il);
			const auto tx = _mm256_permute2f128_ps(tl, th, 0x20);

			const auto near2 = _mm256_set1_ps(std::bit_cast<float>(0x3fffffff));
			const auto rem = _mm256_mul_ps(_mm256_sub_ps(x, tx), near2);
			dst[1] = _mm256_add_epi64(th, _mm256_cvttps_epi64(_mm256_permute2f128_ps(rem, rem, 0x11)));
			dst[0] = _mm256_add_epi64(tl, _mm256_cvttps_epi64(rem));
#else
			__m256i tmp;
			round<std::int32_t>(x, &tmp);
			const auto ih = _mm256_unpackhi_ps(std::bit_cast<__m256>(tmp), _mm256_setzero_ps());
			const auto il = _mm256_unpacklo_ps(std::bit_cast<__m256>(tmp), _mm256_setzero_ps());
			dst[0] = std::bit_cast<__m256i>(_mm256_permute2f128_ps(il, ih, 0x20));
			dst[1] = std::bit_cast<__m256i>(_mm256_permute2f128_ps(il, ih, 0x31));
#endif
		}
		template<std::same_as<float> T>
		DPM_FORCEINLINE void round(__m256 x, __m256 *dst) noexcept
		{
			const auto sign = bit_and(x, _mm256_set1_ps(-0.0f));
			const auto off = bit_or(sign, std::bit_cast<__m256>(_mm256_set1_epi32(0x3effffff)));
			*dst = _mm256_round_ps(bit_and(x, off), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
		}

		template<integral_of_size<4> T>
		DPM_FORCEINLINE void rint(__m256 x, __m256i *dst) noexcept { *dst = _mm256_cvtps_epi32(x); }
		template<integral_of_size<8> T>
		DPM_FORCEINLINE void rint(__m256 x, __m256i *dst) noexcept
		{
			const auto tmp = _mm256_cvtps_epi32(x);
			const auto ih = _mm256_unpackhi_ps(std::bit_cast<__m256>(tmp), _mm256_setzero_ps());
			const auto il = _mm256_unpacklo_ps(std::bit_cast<__m256>(tmp), _mm256_setzero_ps());
			dst[0] = std::bit_cast<__m256i>(_mm256_permute2f128_ps(il, ih, 0x20));
			dst[1] = std::bit_cast<__m256i>(_mm256_permute2f128_ps(il, ih, 0x31));
		}
		template<std::same_as<float> T>
		DPM_FORCEINLINE void rint(__m256 x, __m256 *dst) noexcept { *dst = _mm256_round_ps(x, _MM_FROUND_RAISE_EXC | _MM_FROUND_CUR_DIRECTION); }
#endif
	}
}

#endif