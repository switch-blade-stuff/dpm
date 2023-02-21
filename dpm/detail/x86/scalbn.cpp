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

#include "except.hpp"

namespace dpm::detail
{
	template<typename T, typename V, typename Vi, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_scalbn(V x, Vi n) noexcept
	{
		const auto ix = std::bit_cast<Vi>(x);
		const auto is_zero = std::bit_cast<Vi>(cmp_eq<T>(x, setzero<V>()));
		auto x_exp = bit_and(bit_shiftr<I, mant_bits<I>>(ix), fill<Vi>(exp_mask<I>));

		/* Subnormal x. */
		const auto sx = mul<T>(x, fill<V>(exp_mult<T>));
		const auto si = std::bit_cast<Vi>(sx);
		auto sk = bit_and(bit_shiftr<I, mant_bits<I>>(si), fill<Vi>(exp_mask<I>));
		sk = add<I>(sk, fill<Vi>(mant_bits<I> + 2));

		/* x_exp = x_exp == 0 ? y_exp : x_exp */
		x_exp = bit_or(x_exp, bit_and(cmp_eq<I>(x_exp, setzero<Vi>()), sk));
		auto y_exp = add<I>(x_exp, n);

		/* Normalize x_exp */
		const auto is_denorm = cmp_gt<I>(y_exp, setzero<Vi>());
		y_exp = add<I>(y_exp, bit_andnot(is_denorm, fill<Vi>(mant_bits<I> + 2)));

#ifdef DPM_PROPAGATE_NAN
		const auto not_fin = std::bit_cast<V>(cmp_eq<I>(x_exp, fill<Vi>(exp_mask<I>)));
#endif
#ifdef DPM_HANDLE_ERRORS
		const auto has_overflow = bit_or(cmp_gt<I>(n, fill<Vi>(max_scalbn<I>)), cmp_gt<I>(y_exp, fill<Vi>(exp_mask<I> + mant_bits<I> + 1)));
		const auto has_underflow = bit_or(cmp_gt<I>(fill<Vi>(-max_scalbn<I>), n), cmp_gt<I>(fill<Vi>(I{1}), y_exp));
#endif

		/* Apply the new exponent & normalize. */
		const auto exp_zeros = fill<Vi>(~(exp_mask<I> << mant_bits<I>));
		y_exp = bit_andnot(is_zero, bit_shiftl<I, mant_bits<I>>(y_exp));
		auto y = std::bit_cast<V>(bit_or(bit_and(ix, exp_zeros), y_exp));
		y = blendv<T>(y, mul<T>(y, fill<V>(exp_multm<T>)), std::bit_cast<V>(is_denorm));

#ifdef DPM_PROPAGATE_NAN
		y = blendv<T>(y, add<T>(x, x), not_fin);
#endif
#ifdef DPM_HANDLE_ERRORS
		const auto x_sign = masksign<T>(x);
		if (test_mask(has_overflow)) [[unlikely]] y = except_oflow<T>(y, x_sign, std::bit_cast<V>(has_overflow));
		if (test_mask(has_underflow)) [[unlikely]] y = except_uflow<T>(y, x_sign, std::bit_cast<V>(has_underflow));
#endif
		return y;
	}

	__m128 DPM_MATHFUNC scalbn(__m128 x, __m128i exp) noexcept { return impl_scalbn<float>(x, exp); }
	__m128d DPM_MATHFUNC scalbn(__m128d x, __m128i exp) noexcept { return impl_scalbn<double>(x, exp); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC scalbn(__m256 x, __m256i exp) noexcept { return impl_scalbn<float>(x, exp); }
	__m256d DPM_MATHFUNC scalbn(__m256d x, __m256i exp) noexcept { return impl_scalbn<double>(x, exp); }
#endif
}

#endif