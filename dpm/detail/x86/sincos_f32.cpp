/*
 * Created by switchblade on 2023-02-01.
 */

#include "../const_f32.hpp"
#include "sincos.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include "polevl.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
	DPM_FORCEINLINE static std::tuple<__m128, __m128, __m128> prepare_sincos(__m128 x, __m128 abs_x) noexcept
	{
		/* y = |x| * 4 / Pi */
		auto y = _mm_mul_ps(abs_x, _mm_set1_ps(fopi_f32));

		/* i = isodd(y) ? y + 1 : y */
		auto i = _mm_cvtps_epi32(y);
		i = _mm_add_epi32(i, _mm_set1_epi32(1));
		i = _mm_and_si128(i, _mm_set1_epi32(~1ll));
		y = _mm_cvtepi32_ps(i); /* y = i */

		/* Extract sign bit mask */
		const auto flip_sign = _mm_slli_epi32(_mm_and_si128(i, _mm_set1_epi32(4)), 29);
		const auto sign = _mm_xor_ps(masksign(x), std::bit_cast<__m128>(flip_sign));

		/* Find polynomial selection mask */
		const auto p_mask = std::bit_cast<__m128>(_mm_and_si128(i, _mm_set1_epi32(2)));
		return {y, sign, _mm_cmpeq_ps(p_mask, _mm_setzero_ps())};
	}

#if defined(DPM_HAS_FMA) || defined(DPM_DYNAMIC_DISPATCH)
	template<sincos_op Mask>
	inline static auto DPM_MATHFUNC("fma") sincos_fma(__m128 x, __m128 abs_x, [[maybe_unused]] __m128 nan_mask, [[maybe_unused]] __m128 zero_mask) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = _mm_fnmadd_ps(y, _mm_set1_ps(dp_sincos_f32[0]), x);
		z = _mm_fnmadd_ps(y, _mm_set1_ps(dp_sincos_f32[1]), z);
		z = _mm_fnmadd_ps(y, _mm_set1_ps(dp_sincos_f32[2]), z);
		const auto zz = _mm_mul_ps(z, z);

		/* p1 (0 <= a <= Pi/4) */
		auto p1 = polevl_fma(zz, std::span{sincof_f32});    /* p1 = sincof_f32(zz) */
		p1 = _mm_fmadd_ps(_mm_mul_ps(p1, zz), z, z);        /* p1 = p1 * zz * z + z */

		/* p2 (Pi/4 <= a <= 0) */
		auto p2 = polevl_fma(zz, std::span{coscof_f32});    /* p2 = coscof_f32(zz) */
		p2 = _mm_mul_ps(_mm_mul_ps(zz, p2), zz);            /* p2 = zz * p2 * zz */
		p2 = _mm_fmadd_ps(zz, _mm_set1_ps(0.5), p2);        /* p2 = zz * 0.5 + p2 */
		p2 = _mm_sub_ps(_mm_set1_ps(1.0), p2);              /* p2 = 1.0 - p2 */

		__m128 p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm_blendv_ps(p2, p1, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm_xor_ps(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_sin = _mm_blendv_ps(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm_blendv_ps(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm_blendv_ps(p1, p2, p_mask);  /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm_xor_ps(p_cos, sign);        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_cos = nan_mask ? NaN : p_cos */
			const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_cos = _mm_blendv_ps(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_cos = zero_mask ? 1.0 : p_cos */
			p_cos = _mm_blendv_ps(p_cos, _mm_set1_ps(1.0), zero_mask);
#endif
		}

		return return_sincos<Mask>(p_sin, p_cos);
	}
#endif

#if defined(DPM_HAS_SSE4_1) || defined(DPM_DYNAMIC_DISPATCH)
	template<sincos_op Mask>
	inline static auto DPM_MATHFUNC("sse4.1") sincos_sse4_1(__m128 x, __m128 abs_x, [[maybe_unused]] __m128 nan_mask, [[maybe_unused]] __m128 zero_mask) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = fnmadd_sse(y, _mm_set1_ps(dp_sincos_f32[0]), x);
		z = fnmadd_sse(y, _mm_set1_ps(dp_sincos_f32[1]), z);
		z = fnmadd_sse(y, _mm_set1_ps(dp_sincos_f32[2]), z);
		const auto zz = _mm_mul_ps(z, z);

		/* p1 */
		auto p1 = polevl_sse(zz, std::span{sincof_f32});    /* p1 = sincof_f32(zz) */
		p1 = fmadd_sse(_mm_mul_ps(p1, zz), z, z);           /* p1 = p1 * zz * z + z */

		/* p2 */
		auto p2 = polevl_sse(zz, std::span{coscof_f32});    /* p2 = coscof_f32(zz) */
		p2 = _mm_mul_ps(_mm_mul_ps(zz, p2), zz);            /* p2 = zz * p2 * zz */
		p2 = fmadd_sse(zz, _mm_set1_ps(0.5), p2);           /* p2 = zz * 0.5 + p2 */
		p2 = _mm_sub_ps(_mm_set1_ps(1.0), p2);              /* p2 = 1.0 - p2 */

		__m128 p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm_blendv_ps(p1, p2, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm_xor_ps(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_sin = _mm_blendv_ps(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm_blendv_ps(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm_or_ps(_mm_andnot_ps(p_mask, p2), _mm_and_ps(p_mask, p1));   /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm_xor_ps(p_cos, sign);                                        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_cos = _mm_blendv_ps(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? 1.0 : p_sin */
			p_cos = _mm_blendv_ps(p_cos, _mm_set1_ps(1.0), zero_mask);
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
	}
#endif

	template<sincos_op Mask>
	inline static auto DPM_MATHFUNC("sse2") sincos_sse(__m128 x, __m128 abs_x, [[maybe_unused]] __m128 nan_mask, [[maybe_unused]] __m128 zero_mask) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = _mm_sub_ps(x, _mm_mul_ps(y, _mm_set1_ps(dp_sincos_f32[0]))); /* _mm_fnmadd_ps */
		z = _mm_sub_ps(z, _mm_mul_ps(y, _mm_set1_ps(dp_sincos_f32[1])));      /* _mm_fnmadd_ps */
		z = _mm_sub_ps(z, _mm_mul_ps(y, _mm_set1_ps(dp_sincos_f32[2])));      /* _mm_fnmadd_ps */
		const auto zz = _mm_mul_ps(z, z);

		/* p1 */
		auto p1 = polevl_sse(zz, std::span{sincof_f32});        /* p1 = sincof_f32(zz) */
		p1 = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(p1, zz), z), z);  /* p1 = p1 * zz * z + z */

		/* p2 */
		auto p2 = polevl_sse(zz, std::span{coscof_f32});        /* p2 = coscof_f32(zz) */
		p2 = _mm_mul_ps(_mm_mul_ps(zz, p2), zz);                /* p2 = zz * p2 * zz */
		p2 = _mm_add_ps(_mm_mul_ps(zz, _mm_set1_ps(0.5)), p2);  /* p2 = zz * 0.5 + p2 */
		p2 = _mm_sub_ps(_mm_set1_ps(1.0), p2);                  /* p2 = 1.0 - p2 */

		__m128 p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm_or_ps(_mm_andnot_ps(p_mask, p1), _mm_and_ps(p_mask, p2));   /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm_xor_ps(p_sin, sign);                                        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_sin = _mm_or_ps(_mm_andnot_ps(nan_mask, p_sin), _mm_and_ps(nan_mask, nan));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm_or_ps(_mm_andnot_ps(zero_mask, p_sin), _mm_and_ps(zero_mask, x));
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm_or_ps(_mm_andnot_ps(p_mask, p2), _mm_and_ps(p_mask, p1));   /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm_xor_ps(p_cos, sign);                                        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_cos = _mm_or_ps(_mm_andnot_ps(nan_mask, p_cos), _mm_and_ps(nan_mask, nan));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? 1.0 : p_sin */
			p_cos = _mm_or_ps(_mm_andnot_ps(zero_mask, p_cos), _mm_and_ps(zero_mask, _mm_set1_ps(1.0)));
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
	}

	std::pair<__m128, __m128> DPM_PUBLIC DPM_MATHFUNC("sse2") sincos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m128 DPM_PUBLIC DPM_MATHFUNC("sse2") sin(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x); }
	__m128 DPM_PUBLIC DPM_MATHFUNC("sse2") cos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x); }

#ifdef DPM_HAS_AVX
#if defined(DPM_HAS_FMA) || defined(DPM_DYNAMIC_DISPATCH)
	DPM_FORCEINLINE static std::tuple<__m256, __m256, __m256> prepare_sincos(__m256 x, __m256 abs_x) noexcept
	{
		/* y = |x| * 4 / Pi */
		auto y = _mm256_mul_ps(abs_x, _mm256_set1_ps(fopi_f32));

		/* i = isodd(y) ? y + 1 : y */
		auto il = _mm_cvtps_epi32(_mm256_extractf128_si256(y, 0));
		auto ih = _mm_cvtps_epi32(_mm256_extractf128_si256(y, 1));
		il = _mm_add_epi64(il, _mm_set1_epi32(1));
		ih = _mm_add_epi64(ih, _mm_set1_epi32(1));
		il = _mm_and_si128(il, _mm_set1_epi32(~1ll));
		ih = _mm_and_si128(ih, _mm_set1_epi32(~1ll));
		y = _mm256_set_m128(_mm_cvtepi32_ps(ih), _mm_cvtepi32_ps(il)); /* y = i */
		const auto i = _mm256_set_m128i(ih, il);

		/* Extract sign bit mask */
		const auto flip_sign = mux_128x2<__m256i>([](auto i) { return _mm_slli_epi32(_mm_and_si128(i, _mm_set1_epi32(4)), 29); }, i);
		const auto sign = _mm256_xor_ps(masksign(x), std::bit_cast<__m256>(flip_sign));

		/* Find polynomial selection mask */
		const auto p_mask = _mm256_and_ps(std::bit_cast<__m256>(i), std::bit_cast<__m256>(_mm256_set1_epi32(2)));
		return {y, sign, _mm256_cmp_ps(p_mask, _mm256_setzero_ps(), _CMP_EQ_OQ)};
	}

	template<sincos_op Mask>
	inline static auto DPM_MATHFUNC("fma") sincos_fma(__m256 x, __m256 abs_x, [[maybe_unused]] __m256 nan_mask, [[maybe_unused]] __m256 zero_mask) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = _mm256_fnmadd_ps(y, _mm256_set1_ps(dp_sincos_f32[0]), x);
		z = _mm256_fnmadd_ps(y, _mm256_set1_ps(dp_sincos_f32[1]), z);
		z = _mm256_fnmadd_ps(y, _mm256_set1_ps(dp_sincos_f32[2]), z);
		const auto zz = _mm256_mul_ps(z, z);

		/* p1 (0 <= a <= Pi/4) */
		auto p1 = polevl_fma(zz, std::span{sincof_f32});    /* p1 = sincof_f32(zz) */
		p1 = _mm256_fmadd_ps(_mm256_mul_ps(p1, zz), z, z);  /* p1 = p1 * zz * z + z */

		/* p2 (Pi/4 <= a <= 0) */
		auto p2 = polevl_fma(zz, std::span{coscof_f32});    /* p2 = coscof_f32(zz) */
		p2 = _mm256_mul_ps(_mm256_mul_ps(zz, p2), zz);      /* p2 = zz * p2 * zz */
		p2 = _mm256_fmadd_ps(zz, _mm256_set1_ps(0.5), p2);  /* p2 = zz * 0.5 + p2 */
		p2 = _mm256_sub_ps(_mm256_set1_ps(1.0), p2);        /* p2 = 1.0 - p2 */

		__m256 p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm256_blendv_ps(p2, p1, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm256_xor_ps(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_sin = _mm256_blendv_ps(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm256_blendv_ps(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm256_blendv_ps(p1, p2, p_mask);  /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm256_xor_ps(p_cos, sign);        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_cos = nan_mask ? NaN : p_cos */
			const auto nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_cos = _mm256_blendv_ps(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_cos = zero_mask ? 1.0 : p_cos */
			p_cos = _mm256_blendv_ps(p_cos, _mm256_set1_ps(1.0), zero_mask);
#endif
		}

		return return_sincos<Mask>(p_sin, p_cos);
	}
#endif

#if defined(DPM_HAS_AVX2) || defined(DPM_DYNAMIC_DISPATCH)
	template<sincos_op Mask>
	inline static auto DPM_MATHFUNC("avx2") sincos_avx2(__m256 x, __m256 abs_x, [[maybe_unused]] __m256 nan_mask, [[maybe_unused]] __m256 zero_mask) noexcept
	{
		/* y = |x| * 4 / Pi */
		auto y = _mm256_mul_ps(abs_x, _mm256_set1_ps(fopi_f32));

		/* i = isodd(y) ? y + 1 : y */
		auto i = _mm256_cvtps_epi32(y);
		i = _mm256_add_epi32(i, _mm256_set1_epi32(1));
		i = _mm256_and_si256(i, _mm256_set1_epi32(~1ll));
		y = _mm256_cvtepi32_ps(i); /* y = i */

		/* Extract sign bit mask */
		const auto flip_sign = _mm256_slli_epi32(_mm256_and_si256(i, _mm256_set1_epi32(4)), 29);
		const auto sign = _mm256_xor_ps(masksign(x), std::bit_cast<__m256>(flip_sign));

		/* Find polynomial selection mask */
		auto p_mask = std::bit_cast<__m256>(_mm256_and_si256(i, _mm256_set1_epi32(2)));
		p_mask = _mm256_cmp_ps(p_mask, _mm256_setzero_ps(), _CMP_EQ_OQ);

		auto z = fnmadd_avx(y, _mm256_set1_ps(dp_sincos_f32[0]), x);
		z = fnmadd_avx(y, _mm256_set1_ps(dp_sincos_f32[1]), z);
		z = fnmadd_avx(y, _mm256_set1_ps(dp_sincos_f32[2]), z);
		const auto zz = _mm256_mul_ps(z, z);

		/* p1 */
		auto p1 = polevl_avx(zz, std::span{sincof_f32});    /* p1 = sincof_f32(zz) */
		p1 = fmadd_avx(_mm256_mul_ps(p1, zz), z, z);        /* p1 = p1 * zz * z + z */

		/* p2 */
		auto p2 = polevl_avx(zz, std::span{coscof_f32});    /* p2 = coscof_f32(zz) */
		p2 = _mm256_mul_ps(_mm256_mul_ps(zz, p2), zz);      /* p2 = zz * p2 * zz */
		p2 = fmadd_avx(zz, _mm256_set1_ps(0.5), p2);        /* p2 = zz * 0.5 + p2 */
		p2 = _mm256_sub_ps(_mm256_set1_ps(1.0), p2);        /* p2 = 1.0 - p2 */

		__m256 p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm256_blendv_ps(p1, p2, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm256_xor_ps(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_sin = _mm256_blendv_ps(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm256_blendv_ps(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm256_or_ps(_mm256_andnot_ps(p_mask, p2), _mm256_and_ps(p_mask, p1));   /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm256_xor_ps(p_cos, sign);                                              /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_cos = _mm256_blendv_ps(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? 1.0 : p_sin */
			p_cos = _mm256_blendv_ps(p_cos, _mm256_set1_ps(1.0), zero_mask);
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
	}
#endif

	template<sincos_op Mask>
	inline static auto DPM_MATHFUNC("avx") sincos_avx(__m256 x, __m256 abs_x, [[maybe_unused]] __m256 nan_mask, [[maybe_unused]] __m256 zero_mask) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = fnmadd_avx(y, _mm256_set1_ps(dp_sincos_f32[0]), x);
		z = fnmadd_avx(y, _mm256_set1_ps(dp_sincos_f32[1]), z);
		z = fnmadd_avx(y, _mm256_set1_ps(dp_sincos_f32[2]), z);
		const auto zz = _mm256_mul_ps(z, z);

		/* p1 */
		auto p1 = polevl_avx(zz, std::span{sincof_f32});    /* p1 = sincof_f32(zz) */
		p1 = fmadd_avx(_mm256_mul_ps(p1, zz), z, z);        /* p1 = p1 * zz * z + z */

		/* p2 */
		auto p2 = polevl_avx(zz, std::span{coscof_f32});    /* p2 = coscof_f32(zz) */
		p2 = _mm256_mul_ps(_mm256_mul_ps(zz, p2), zz);      /* p2 = zz * p2 * zz */
		p2 = fmadd_avx(zz, _mm256_set1_ps(0.5), p2);        /* p2 = zz * 0.5 + p2 */
		p2 = _mm256_sub_ps(_mm256_set1_ps(1.0), p2);        /* p2 = 1.0 - p2 */

		__m256 p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm256_blendv_ps(p1, p2, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm256_xor_ps(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_sin = _mm256_blendv_ps(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm256_blendv_ps(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm256_or_ps(_mm256_andnot_ps(p_mask, p2), _mm256_and_ps(p_mask, p1));   /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm256_xor_ps(p_cos, sign);                                              /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());
			p_cos = _mm256_blendv_ps(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? 1.0 : p_sin */
			p_cos = _mm256_blendv_ps(p_cos, _mm256_set1_ps(1.0), zero_mask);
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
	}

	std::pair<__m256, __m256> DPM_PUBLIC DPM_MATHFUNC("avx") sincos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m256 DPM_PUBLIC DPM_MATHFUNC("avx") sin(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x); }
	__m256 DPM_PUBLIC DPM_MATHFUNC("avx") cos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x); }
#endif
}

#endif