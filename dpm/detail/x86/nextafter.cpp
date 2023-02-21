/*
 * Created by switchblade on 2023-02-20.
 */

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_nextafter(V a, V b)
	{
		auto ia = std::bit_cast<Vi>(a);
		auto ib = std::bit_cast<Vi>(b);
		const auto a_sign = std::bit_cast<Vi>(bit_and(a, fill<V>(sign_bit<T>)));
		const auto b_sign = std::bit_cast<Vi>(bit_and(b, fill<V>(sign_bit<T>)));
		const auto abs_a = bit_xor(ia, a_sign);
		const auto abs_b = bit_xor(ib, b_sign);

		const auto zero_mask = cmp_eq<I>(abs_a, setzero<Vi>());
		/* x_off = (abs_a > abs_b || (a_sign ^ b_sign)) ? -1 : 1 */
		const auto sub_mask = bit_or(cmp_gt<I>(setzero<Vi>(), bit_xor(a_sign, b_sign)), cmp_gt<I>(abs_a, abs_b));
		const auto x_off = bit_or(fill<Vi>(I{1}), sub_mask);
		/* ix = ax == 0 ? y_sign | 1 : ix + x_off */
		ia = blendv<I>(add<I>(ia, x_off), bit_or(b_sign, fill<Vi>(I{1})), zero_mask);

		/* raise overflow if ix is infinite and a is finite & return NaN if any is NaN. */
		const auto inf_exp = fill<Vi>(exp_mask<I> << mant_bits<I>);
		const auto exp = bit_and(ia, inf_exp);
		auto c = std::bit_cast<V>(ia);

		/* Check domain & propagate NaN */
		const auto eq_mask = cmp_eq<T>(a, b);
#ifdef DPM_PROPAGATE_NAN
		/* c = isnan(a) || isnan(b) ? a | b : c */
		const auto nan_mask = isunord(a, b);
		c = blendv<T>(c, bit_or(a, b), nan_mask);
#if defined(DPM_HANDLE_ERRORS) && math_errhandling
		int error = 0;
		/* Raise overflow if exp == inf */
		const auto oflow_mask = std::bit_cast<V>(cmp_eq<I>(exp, inf_exp));
		if (test_mask(bit_andnot(nan_mask, oflow_mask))) [[unlikely]] error |= FE_OVERFLOW;
		/* Raise underflow if exp == 0 */
		const auto uflow_mask = std::bit_cast<V>(cmp_eq<I>(exp, setzero<Vi>()));
		if (test_mask(bit_andnot(eq_mask, uflow_mask))) [[unlikely]] error |= FE_UNDERFLOW;

		/* Cannot use except_oflow & except_uflow, as we would have to discard the results. */
		if (error != 0) [[unlikely]]
		{
#if math_errhandling & MATH_ERREXCEPT
			std::feraiseexcept(error);
#endif
#if math_errhandling & MATH_ERRNO
			errno = ERANGE;
#endif
		}
#endif
#endif
		/* return a == b ? b : c */
		return blendv<T>(c, b, eq_mask);
	}

	__m128 DPM_MATHFUNC nextafter(__m128 from, __m128 to) noexcept { return impl_nextafter<float>(from, to); }
	__m128d DPM_MATHFUNC nextafter(__m128d from, __m128d to) noexcept { return impl_nextafter<double>(from, to); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC nextafter(__m256 from, __m256 to) noexcept { return impl_nextafter<float>(from, to); }
	__m256d DPM_MATHFUNC nextafter(__m256d from, __m256d to) noexcept { return impl_nextafter<double>(from, to); }
#endif
}

#endif