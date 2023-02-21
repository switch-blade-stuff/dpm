/*
 * Created by switchblade on 2023-02-06.
 */

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V impl_atan(V x) noexcept
	{
		const auto v_pio4 = fill<V>(pio4<T>);
		const auto v_pio2 = fill<V>(pio2<T>);
		const auto v_mone = fill<V>(-one<T>);
		const auto v_one = fill<V>(one<T>);
		const auto x_sign = masksign<T>(x);
		auto abs_x = bit_xor(x, x_sign);

		/* pio2_mask = |x| > tan(3 * pi / 8) */
		const auto pio2_mask = cmp_gt<T>(abs_x, fill<V>(tan3pio8<T>));
		/* pio4_mask = |x| > 0.66 */
		const auto pio4_mask = cmp_gt<T>(abs_x, fill<V>(p66<T>));

		/* Range reduction. */
		auto y = setzero<V>();
		const auto x_zero = cmp_eq<T>(abs_x, setzero<V>()); /* Avoid -1.0 / 0 */
		const auto x_pio4 = div<T>(add<T>(abs_x, v_mone), add<T>(abs_x, v_one));
		const auto x_pio2 = div<T>(v_mone, blendv<T>(abs_x, v_one, x_zero));
		abs_x = blendv<T>(abs_x, x_pio4, pio4_mask);
		abs_x = blendv<T>(abs_x, x_pio2, pio2_mask);
		y = blendv<T>(y, v_pio4, pio4_mask);
		y = blendv<T>(y, v_pio2, pio2_mask);

		auto z = mul<T>(abs_x, abs_x);
		const auto p0 = polevl(z, std::span{atan_p<T>});
		const auto p1 = polevl(z, std::span{atan_q<T>});
		z = mul<T>(z, div<T>(p0, p1));
		z = fmadd(abs_x, z, abs_x);

		/* Offset z by asin_off<T> * 0.5 once if |x| > 0.66, or twice if |x| > tan(3 * pi / 8). */
		const auto z_off = fill<V>(asin_off<T> * half<T>);
		z = add<T>(z, bit_and(bit_or(pio2_mask, pio4_mask), z_off));
		z = add<T>(z, bit_and(pio2_mask, z_off));
		y = add<T>(y, z);

		/* atan(+-inf) = +-Pi/2. */
		const auto inf_mask = isinf_abs(abs_x);
		y = blendv<T>(y, v_pio2, inf_mask);
		/* Restore sign. */
		return bit_xor(y, x_sign);
	}

	[[nodiscard]] __m128 DPM_MATHFUNC atan(__m128 x) noexcept { return impl_atan<float>(x); }
	[[nodiscard]] __m128d DPM_MATHFUNC atan(__m128d x) noexcept { return impl_atan<double>(x); }

#ifdef DPM_HAS_AVX
	[[nodiscard]] __m256 DPM_MATHFUNC atan(__m256 x) noexcept { return impl_atan<float>(x); }
	[[nodiscard]] __m256d DPM_MATHFUNC atan(__m256d x) noexcept { return impl_atan<double>(x); }
#endif
}

#endif
