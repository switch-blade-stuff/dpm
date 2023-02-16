/*
 * Created by switchblade on 2023-02-06.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "except.hpp"
#include "polevl.hpp"
#include "pow.hpp"

namespace dpm::detail
{
	template<typename T, typename V>
	[[nodiscard]] static V eval_asin(V abs_x, V x_sign) noexcept
	{
		const auto v_pio4 = fill<V>(pio4<T>);

		/* Polynomial selection mask */
		const auto p_mask = cmp_gt<T>(abs_x, fill<V>(five_eights<T>));
		const auto x_mask = cmp_lt<T>(abs_x, fill<V>(asin_pmin<T>));

		/* p1: abs(x) > 0.625 */
		auto a1 = sub<T>(fill<V>(one<T>), abs_x);
		auto p1 = mul<T>(a1, div<T>(polevl(a1, std::span{asin_r<T>}), polevl(a1, std::span{asin_s<T>})));
		a1 = sqrt(add<T>(a1, a1));
		auto b1 = sub<T>(v_pio4, a1);
		a1 = fmsub(a1, p1, fill<V>(asin_off<T>));
		p1 = add<T>(sub<T>(b1, a1), v_pio4);

		/* p2: abs(x) <= 0.625 */
		const auto a2 = mul<T>(abs_x, abs_x);
		auto p2 = div<T>(polevl(a2, std::span{asin_p<T>}), polevl(a2, std::span{asin_q<T>}));
		p2 = fmadd(abs_x, mul<T>(a2, p2), abs_x);

		/* Select between p1 and p2 & restore sign */
		auto p = blendv<T>(p2, p1, p_mask); /* p = p_mask ? p1 : p2 */
		p = blendv<T>(p, abs_x, x_mask);    /* p = abs(x) < asin_min ? abs(x) : p */
		return bit_xor(p, x_sign);          /* return sign ? -p : p */
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V impl_asin(V x) noexcept
	{
		const auto x_sign = masksign<T>(x);
		auto abs_x = bit_xor(x, x_sign);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = cmp_gt<T>(abs_x, fill<V>(one<T>)); test_mask(m))
			[[unlikely]] abs_x = except_invalid<T>(abs_x, m);
#endif
		return eval_asin<T>(abs_x, x_sign);
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V impl_acos(V x) noexcept
	{
		const auto v_pio4 = fill<V>(pio4<T>);
		const auto x_sign = masksign<T>(x);
		auto abs_x = bit_xor(x, x_sign);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = cmp_gt<T>(abs_x, fill<V>(one<T>)); test_mask(m))
			[[unlikely]] abs_x = except_invalid<T>(abs_x, m);
#endif
		/* c_mask = x > 0.5 */
		const auto c_mask = cmp_gt<T>(x, fill<V>(half<T>));
		/* acos1: x > 0.5 */
		const auto acos1 = mul<T>(fill<V>(two<T>), eval_asin<T>(sqrt(fmadd(abs_x, fill<V>(-half<T>), fill<V>(half<T>))), setzero<V>()));
		/* acos2: x <= 0.5 */
		const auto acos2 = add<T>(add<T>(sub<T>(v_pio4, eval_asin<T>(abs_x, x_sign)), fill<V>(asin_off<T>)), v_pio4);
		/* Select result. */
		return blendv<T>(acos2, acos1, c_mask);
	}

	__m128 DPM_MATHFUNC asin(__m128 x) noexcept { return impl_asin<float>(x); }
	__m128 DPM_MATHFUNC acos(__m128 x) noexcept { return impl_acos<float>(x); }
	__m128d DPM_MATHFUNC asin(__m128d x) noexcept { return impl_asin<double>(x); }
	__m128d DPM_MATHFUNC acos(__m128d x) noexcept { return impl_acos<double>(x); }

#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC asin(__m256 x) noexcept { return impl_asin<float>(x); }
	__m256 DPM_MATHFUNC acos(__m256 x) noexcept { return impl_acos<float>(x); }
	__m256d DPM_MATHFUNC asin(__m256d x) noexcept { return impl_asin<double>(x); }
	__m256d DPM_MATHFUNC acos(__m256d x) noexcept { return impl_acos<double>(x); }
#endif
}

#endif
