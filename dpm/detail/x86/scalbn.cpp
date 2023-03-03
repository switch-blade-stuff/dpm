/*
 * Created by switchblade on 2023-02-20.
 */

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "except.hpp"

namespace dpm::detail
{
	template<typename T, typename V, typename Vi, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_scalbn(V x, Vi n) noexcept
	{
		const auto ix = std::bit_cast<Vi>(x);
		const auto x_exp = bit_and(bit_shiftr<I, mant_bits<I>>(ix), fill<Vi>(exp_mask<I>));

		/* Subnormal x. */
		const auto norm_x = mul<T>(x, fill<V>(exp_mult<T>));
		const auto norm_i = std::bit_cast<Vi>(norm_x);
		auto norm_exp = bit_and(bit_shiftr<I, mant_bits<I>>(norm_i), fill<Vi>(exp_mask<I>));
		norm_exp = sub<I>(norm_exp, fill<Vi>(mant_bits<I> + 2));

		/* norm_exp = x_exp == 0 ? norm_exp : x_exp */
		const auto subnorm_mask = cmp_eq<I>(x_exp, setzero<Vi>());
		norm_exp = bit_or(x_exp, bit_and(subnorm_mask, norm_exp));

		/* Handle non-finite x before modifying norm_exp */
#ifdef DPM_PROPAGATE_NAN
		const auto not_fin = cmp_eq<I>(norm_exp, fill<Vi>(exp_mask<I>));
#endif
		/* Input is normalized, apply n to exponent. */
		norm_exp = add<I>(norm_exp, n);

		/* Handle subnormal result for norm_exp <= 0. */
		const auto norm_result = cmp_gt<I>(norm_exp, setzero<Vi>());
		norm_exp = add<I>(norm_exp, bit_andnot(norm_result, fill<Vi>(mant_bits<I> + 2)));

#ifdef DPM_HANDLE_ERRORS
		const auto has_overflow = bit_andnot(not_fin, bit_or(cmp_gt<I>(n, fill<Vi>(max_scalbn<I>)), cmp_gt<I>(norm_exp, fill<Vi>(exp_mask<I> - 1))));
		const auto has_underflow = bit_andnot(not_fin, bit_or(cmp_gt<I>(fill<Vi>(-max_scalbn<I>), n), cmp_gt<I>(fill<Vi>(I{1}), norm_exp)));
#endif

		/* Apply the new exponent & normalize. */
		norm_exp = bit_shiftl<I, mant_bits<I>>(norm_exp);
		const auto exp_zeros = fill<Vi>(~(exp_mask<I> << mant_bits<I>));
		auto y = std::bit_cast<V>(bit_or(bit_and(ix, exp_zeros), norm_exp));

		/* Denormalize result if needed. */
		y = blendv<T>(mul<T>(y, fill<V>(exp_multm<T>)), y, std::bit_cast<V>(norm_result));
		/* If either x or exponent are 0, return x. */
		auto fwd_mask = bit_or(cmp_eq<T>(x, setzero<V>()), std::bit_cast<V>(cmp_eq<I>(n, setzero<Vi>())));
#ifdef DPM_PROPAGATE_NAN
		fwd_mask = bit_or(fwd_mask, std::bit_cast<V>(not_fin));
#endif
		y = blendv<T>(y, x, fwd_mask);

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