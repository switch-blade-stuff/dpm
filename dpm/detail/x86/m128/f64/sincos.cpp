/*
 * Created by switchblade on 2023-01-10.
 */

#include "../../../ieee754/const_f64.hpp"
#include "../../../dispatch.hpp"
#include "../../cpuid.hpp"
#include "../sign_ops.hpp"
#include "../isnan.hpp"
#include "utility.hpp"
#include "sincos.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm::detail
{
#if defined(DPM_HAS_FMA) || defined(DPM_DYNAMIC_DISPATCH)
	/* TODO: Implement FMA versions. */
	[[maybe_unused]] static void DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("fma") sincos_fma(__m128d, __m128d &, __m128d &) noexcept {}
	[[maybe_unused]] static __m128d DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("fma") sin_fma(__m128d x) noexcept { return x; }
	[[maybe_unused]] static __m128d DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("fma") cos_fma(__m128d x) noexcept { return x; }
#endif

#if defined(DPM_HAS_SSE4_1) || defined(DPM_DYNAMIC_DISPATCH)
	/* TODO: Implement SSE4.1 versions. */
	[[maybe_unused]] static void DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("sse4.1") sincos_sse4_1(__m128d, __m128d &, __m128d &) noexcept {}
	[[maybe_unused]] static __m128d DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("sse4.1") sin_sse4_1(__m128d x) noexcept { return x; }
	[[maybe_unused]] static __m128d DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("sse4.1") cos_sse4_1(__m128d x) noexcept { return x; }
#endif

	/* TODO: Implement SSE2 versions. */
	[[maybe_unused]] static void DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("sse2") sincos_sse2(__m128d, __m128d &, __m128d &) noexcept
	{
	}
	[[maybe_unused]] static __m128d DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("sse2") sin_sse2(__m128d x) noexcept
	{
		return _mm_add_pd(x, x);
	}
	[[maybe_unused]] static __m128d DPM_PRIVATE DPM_VECTORCALL DPM_TARGET("sse2") cos_sse2(__m128d x) noexcept
	{
		const auto abs_x = x86_abs(x);
#ifdef DPM_PROPAGATE_NAN
		const auto nan_mask = x86_isnan(x);
#endif
#ifdef DPM_HANDLE_ERRORS
		/* TODO: Check for domain. */
		const auto dom_mask = _mm_setzero_pd();
		if (abs_x > lossth) mtherr("cos", TLOSS);
#endif

		/* y = |x| * 4 / Pi */
		auto y = _mm_mul_pd(abs_x, _mm_set1_pd(fopi_f64));

		/* i = isodd(y) ? y + 1 : y */
		auto i = x86_cvt_f64_i64(y);
		i = _mm_add_epi64(i, _mm_set1_epi64x(1));
		i = _mm_and_si128(i, _mm_set1_epi64x(~1));
		y = x86_cvt_i64_f64(i); /* y = i */

		/* Extract sign bit mask */
		const auto flip_sign = _mm_slli_epi64(_mm_and_si128(i, _mm_set1_epi64x(4)), 61);
		const auto sign = _mm_xor_pd(x86_masksign(x), std::bit_cast<__m128d>(flip_sign));

		/* Extended precision modular arithmetic */
		z = ((x - y * DP1) - y * DP2) - y * DP3;
		zz = z * z;

		if (i & 0b10)
			y = z + z * zz * polevl(zz, sincof, 5);
		else
			y = 1.0 - ldexp(zz, -1) + zz * zz * polevl(zz, coscof, 5);

		/* Restore sign using the bit mask */
		if (sign < 0) y = -y;
		return (y);

#ifdef DPM_HANDLE_ERRORS
		/* TODO: Set errno & floating-point error codes. */
#endif
#ifdef DPM_PROPAGATE_NAN
		/* TODO: Blend with NaN using nan_mask. */
#endif
	}

#ifdef DPM_DYNAMIC_DISPATCH
	void x86_sincos(__m128d x, __m128d &sin, __m128d &cos) noexcept
	{
		constinit static dispatcher sincos_disp = []()
		{
#ifndef DPM_HAS_FMA
			if (!cpuid::has_fma())
			{
#ifndef DPM_HAS_SSE4_1
				if (!cpuid::has_sse4_1())
					return sincos_sse2;
#endif
				return sincos_sse4_1;
			}
			return sincos_fma;
#endif
		};
		sincos_disp(x, sin, cos);
	}
	__m128d x86_sin(__m128d x) noexcept
	{
		constinit static dispatcher sin_disp = []()
		{
#ifndef DPM_HAS_FMA
			if (!cpuid::has_fma())
			{
#ifndef DPM_HAS_SSE4_1
				if (!cpuid::has_sse4_1())
					return sin_sse2;
#endif
				return sin_sse4_1;
			}
			return sin_fma;
#endif
		};
		return sin_disp(x);
	}
	__m128d x86_cos(__m128d x) noexcept
	{
		constinit static dispatcher cos_disp = []()
		{
#ifndef DPM_HAS_FMA
			if (!cpuid::has_fma())
			{
#ifndef DPM_HAS_SSE4_1
				if (!cpuid::has_sse4_1())
					return cos_sse2;
#endif
				return cos_sse4_1;
			}
			return cos_fma;
#endif
		};
		return cos_disp(x);
	}
#else
	void x86_sincos(__m128d x, __m128d &sin, __m128d &cos) noexcept
	{
#if defined(DPM_HAS_FMA)
		sincos_fma(x, sin, cos);
#elif defined(DPM_HAS_SSE4_1)
		sincos_sse4_1(x, sin, cos);
#else
		sincos_sse2(x, sin, cos);
#endif
	}
	__m128d x86_sin(__m128d x) noexcept
	{
#if defined(DPM_HAS_FMA)
		return sin_fma(x);
#elif defined(DPM_HAS_SSE4_1)
		return sin_sse4_1(x);
#else
		return sin_sse2(x);
#endif
	}
	__m128d x86_cos(__m128d x) noexcept
	{
#if defined(DPM_HAS_FMA)
		return cos_fma(x);
#elif defined(DPM_HAS_SSE4_1)
		return cos_sse4_1(x);
#else
		return cos_sse2(x);
#endif
	}
#endif
}

#endif