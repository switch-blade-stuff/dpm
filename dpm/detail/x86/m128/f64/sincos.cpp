/*
 * Created by switchblade on 2023-01-10.
 */

#include "../../../ieee754/const_sincos.hpp"
#include "../../../dispatch.hpp"
#include "../../cpuid.hpp"
#include "../sign_ops.hpp"
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
		return _mm_add_pd(x, x);
	}

	void x86_sincos(__m128d x, __m128d &sin, __m128d &cos) noexcept
	{
#ifdef DPM_DYNAMIC_DISPATCH
		constinit static dispatcher sincos_disp = []()
		{
			if (cpuid::has_avx() && cpuid::has_fma())
				return sincos_fma;
			else if (cpuid::has_sse4_1())
				return sincos_sse4_1;
			else
				return sincos_sse2;
		};
		sincos_disp(x, sin, cos);
#elif defined(DPM_HAS_FMA)
		return sincos_fma(x, sin, cos);
#elif defined(DPM_HAS_SSE4_1)
		return sincos_sse4_1(x, sin, cos);
#else
		return sincos_sse2(x, sin, cos);
#endif
	}
	__m128d x86_sin(__m128d x) noexcept
	{
#ifdef DPM_DYNAMIC_DISPATCH
		constinit static dispatcher sin_disp = []()
		{
			if (cpuid::has_avx() && cpuid::has_fma())
				return sin_fma;
			else if (cpuid::has_sse4_1())
				return sin_sse4_1;
			else
				return sin_sse2;
		};
		return sin_disp(x);
#elif defined(DPM_HAS_FMA)
		return sin_fma(x);
#elif defined(DPM_HAS_SSE4_1)
		return sin_sse4_1(x);
#else
		return sin_sse2(x, sin, cos);
#endif
	}
	__m128d x86_cos(__m128d x) noexcept
	{
#ifdef DPM_DYNAMIC_DISPATCH
		constinit static dispatcher cos_disp = []()
		{
			if (cpuid::has_avx() && cpuid::has_fma())
				return cos_fma;
			else if (cpuid::has_sse4_1())
				return cos_sse4_1;
			else
				return cos_sse2;
		};
		return cos_disp(x);
#elif defined(DPM_HAS_FMA)
		return cos_fma(x);
#elif defined(DPM_HAS_SSE4_1)
		return cos_sse4_1(x);
#else
		return cos_sse2(x, sin, cos);
#endif
	}
}

#endif