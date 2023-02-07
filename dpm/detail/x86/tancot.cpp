/*
 * Created by switchblade on 2023-02-06.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "../fconst.hpp"
#include "polevl.hpp"
#include "power.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
	enum tancot_op { OP_TAN, OP_COT };

	template<std::same_as<double> T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V tancot_poly(V abs_x, V x) noexcept
	{
		x = fmadd(x, fill<V>(dp_tancotd[2]), fmadd(x, fill<V>(dp_tancotd[1]), fmadd(x, fill<V>(dp_tancotd[0]), abs_x)));
		const auto x2 = mul<T>(x, x);
		if (const auto p_mask = cmp_gt<T>(x2, fill<V>(1.0e-14)); movemask<T>(p_mask))
		{
			const auto p0 = mul<T>(polevl(x2, std::span{tancot_pd}), x2);
			const auto p1 = polevl(x2, std::span{tancot_qd});
			const auto px = add<T>(x, mul<T>(x, div<T>(p0, p1)));
			x = blendv<T>(x, px, p_mask);
		}
		return x;
	}
	template<std::same_as<float> T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V tancot_poly(V abs_x, V x) noexcept
	{
		x = fmadd(x, fill<V>(dp_tancotf[2]), fmadd(x, fill<V>(dp_tancotf[1]), fmadd(x, fill<V>(dp_tancotf[0]), abs_x)));
		const auto x2 = mul<T>(x, x);
		if (const auto p_mask = cmp_gt<T>(x2, fill<V>(1.0e-4f)); movemask<T>(p_mask))
		{
			const auto px = fmadd(mul<T>(polevl(x2, std::span{tancot_pf}), x2), x, x);
			x = blendv<T>(x, px, p_mask);
		}
		return x;
	}

	template<typename T, tancot_op Op, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_tancot(V x) noexcept
	{
		constexpr auto extent = sizeof(V) / sizeof(T);
		const auto x_sign = masksign(x);
		const auto abs_x = bit_xor(x, x_sign);

#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
#ifdef DPM_HANDLE_ERRORS
		auto dom_mask = isinf_abs(abs_x);
		if constexpr (Op == OP_COT)
		{
			/* 0.0 is domain error for cotangent */
			dom_mask = bit_or(dom_mask, cmp_eq<T>(abs_x, setzero<V>()));
		}
		if (movemask<T>(dom_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
		const auto nan_mask = bit_or(dom_mask, isunord(x, x));
#else
		const auto nan_mask = isunord(x, x);
#endif
		if (movemask<T>(nan_mask) == fill_bits<extent>()) [[unlikely]]
			return fill<V>(std::numeric_limits<T>::quiet_NaN());
#endif

		/* y = |x| * 4 / Pi */
		auto y = mul<T>(abs_x, fill<V>(fopi<T>));

		/* Set rounding mode to truncation. */
		const auto old_csr = _mm_getcsr();
		_mm_setcsr((old_csr & ~_MM_ROUND_MASK) | _MM_ROUND_TOWARD_ZERO);

		/* i = isodd(y) ? y + 1 : y */
		auto i = cvt<I, T>(y);
		i = add<I>(i, fill<decltype(i)>(I{1}));
		i = bit_and(i, fill<decltype(i)>(I{~1ll}));
		y = cvt<T, I>(i);

		/* Restore mxcsr */
		_mm_setcsr(old_csr);

		/* Calculate result polynomial. */
		y = tancot_poly<T>(abs_x, y);

		/* Adjust result polynomial for tangent or cotangent. */
		const auto j2_mask = cmp_eq<I>(bit_and(i, fill<decltype(i)>(I{2})), setzero<decltype(i)>());
		if constexpr (Op == tancot_op::OP_COT)
		{
			const auto neg_y = bit_xor(y, fill<V>(T{-0.0}));
			y = blendv<T>(neg_y, rcp(y), std::bit_cast<V>(j2_mask));
		}
		else
		{
			/* Ignore reciprocal for tan(0). */
			const auto y_nz = bit_or(y, bit_and(std::bit_cast<V>(j2_mask), fill<V>(T{1})));
			const auto rcp_y = bit_xor(rcp(y_nz), fill<V>(T{-0.0}));
			y = blendv<T>(rcp_y, y, std::bit_cast<V>(j2_mask));
		}

		/* Restore sign. */
		y = bit_xor(y, x_sign);
#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
		y = blendv<T>(y, fill<V>(std::numeric_limits<T>::quiet_NaN()), nan_mask);
#endif
		return y;
	}

	[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC tan(__m128 x) noexcept { return impl_tancot<float, tancot_op::OP_TAN>(x); }
	[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC cot(__m128 x) noexcept { return impl_tancot<float, tancot_op::OP_COT>(x); }
	[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC tan(__m128d x) noexcept { return impl_tancot<double, tancot_op::OP_TAN>(x); }
	[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC cot(__m128d x) noexcept { return impl_tancot<double, tancot_op::OP_COT>(x); }

#ifdef DPM_HAS_AVX
	[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC tan(__m256 x) noexcept { return impl_tancot<float, tancot_op::OP_TAN>(x); }
	[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC cot(__m256 x) noexcept { return impl_tancot<float, tancot_op::OP_COT>(x); }
	[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC tan(__m256d x) noexcept { return impl_tancot<double, tancot_op::OP_TAN>(x); }
	[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC cot(__m256d x) noexcept { return impl_tancot<double, tancot_op::OP_COT>(x); }
#endif
}

#endif