/*
 * Created by switch_blade on 2023-02-10.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "class.hpp"
#include "cvt.hpp"

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_frexp(V x, Vi &out_exp)
	{
		constexpr auto exp_ones = exp_mask<I> << mant_bits<I>;
		auto ix = std::bit_cast<Vi>(x);
		const auto exp_x = bit_and(ix, fill<Vi>(exp_ones));

		const auto not_fin = cmp_eq<I>(exp_x, fill<Vi>(exp_ones));
		const auto is_subn = cmp_eq<I>(exp_x, setzero<Vi>());
		const auto is_zero = cmp_eq<T>(x, setzero<V>());

		/* off = is_subn ? mant_bits + 2 : 0 */
		const auto off = bit_and(is_subn, fill<Vi>(mant_bits<I> + 2));
		auto norm_x = mul<T>(x, fill<V>(exp_mult<T>));
		const auto norm_ix = std::bit_cast<Vi>(norm_x);
		norm_x = blendv<T>(x, norm_x, std::bit_cast<V>(is_subn));
		auto norm_exp = bit_and(std::bit_cast<Vi>(norm_x), fill<Vi>(exp_ones));
		norm_exp = blendv<I>(exp_x, norm_exp, is_subn);
		ix = blendv<I>(ix, norm_ix, is_subn);

		/* norm_x = (ix & ~exp_bits) | exp_middle */
		norm_x = std::bit_cast<V>(bit_or(bit_and(ix, fill<Vi>(~exp_ones)), fill<Vi>(exp_middle<I> << mant_bits<I>)));
		norm_exp = bit_shiftr<I, mant_bits<I>>(norm_exp) - fill<Vi>(exp_off<I> - 1) - off;

		const auto not_fin_or_zero = bit_or(std::bit_cast<Vi>(is_zero), not_fin);
		norm_x = blendv<T>(norm_x, add<T>(x, x), std::bit_cast<V>(not_fin_or_zero));
		out_exp = blendv<I>(norm_exp, exp_x, not_fin_or_zero);
		return norm_x;
	}

	__m128 frexp(__m128 x, __m128i &out_exp) noexcept { return impl_frexp<float>(x, out_exp); }
	__m128d frexp(__m128d x, __m128i &out_exp) noexcept { return impl_frexp<double>(x, out_exp); }
#ifdef DPM_HAS_AVX
	__m256 frexp(__m256 x, __m256i &out_exp) noexcept { return impl_frexp<float>(x, out_exp); }
	__m256d frexp(__m256d x, __m256i &out_exp) noexcept { return impl_frexp<double>(x, out_exp); }
#endif

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
		if (test_mask<V>(has_overflow)) [[unlikely]]
		{
			const auto vhuge = fill<V>(huge<T>);
			y = blendv<T>(y, mul<T>(vhuge, copysign<T>(vhuge, x)), std::bit_cast<V>(has_underflow));
		}
		if (test_mask<V>(has_underflow)) [[unlikely]]
		{
			const auto vtiny = fill<V>(tiny<T>);
			y = blendv<T>(y, mul<T>(vtiny, copysign<T>(vtiny, x)), std::bit_cast<V>(has_underflow));
		}
#endif
		return y;
	}

	__m128 scalbn(__m128 x, __m128i exp) noexcept { return impl_scalbn<float>(x, exp); }
	__m128d scalbn(__m128d x, __m128i exp) noexcept { return impl_scalbn<double>(x, exp); }
#ifdef DPM_HAS_AVX
	__m256 scalbn(__m256 x, __m256i exp) noexcept { return impl_scalbn<float>(x, exp); }
	__m256d scalbn(__m256d x, __m256i exp) noexcept { return impl_scalbn<double>(x, exp); }
#endif

	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_modf(V x, V &ip) noexcept
	{
		const auto ix = std::bit_cast<Vi>(x);

		/* e = ((ix >> mant_bits) & exp_mask) - exp_off */
		auto e = bit_shiftr<I, mant_bits<I>>(ix);
		e = bit_and(e, fill<Vi>(exp_mask<I>));
		e = sub<I>(e, fill<Vi>(exp_off<I>));

		/* e >= mant_bits */
		const auto int_mask = std::bit_cast<V>(cmp_gt<I>(e, fill<Vi>(mant_bits<I> - 1)));
		const auto int_x = bit_and(x, fill<V>(sign_bit<T>));

		/* e < 0 */
		const auto fract_mask = std::bit_cast<V>(cmp_gt<I>(setzero<Vi>(), e));
		const auto fract_ip = bit_and(x, fill<V>(sign_bit<T>));
		const auto fract_x = x;

		/* mask = -1 >> (exp_bits + 1) >> e */
		const auto mask = std::bit_cast<V>(bit_shiftr<I>(bit_shiftr<I, exp_bits<I> + 1>(fill<Vi>(I{-1})), e));

		/* mask_zero = (ix & mask) == 0 */
		auto mask_zero = cmp_eq<T>(bit_and(std::bit_cast<V>(ix), mask), setzero<V>());
		/* mask_zero = mask_zero && !(fract_mask | int_mask) */
		mask_zero = bit_andnot(fract_mask, mask_zero);
		mask_zero = bit_andnot(int_mask, mask_zero);

		const auto x1 = bit_and(x, fill<V>(sign_bit<T>));   /* x1 = x & -0.0 */
		const auto x2 = bit_andnot(mask, x);                /* x2 = x & ~mask */

		/* ip = e < 0 ? x : x & -0.0 */
		ip = blendv<T>(x, fract_ip, fract_mask);
		/* ip = ix & mask ? x2 : ip */
		ip = blendv<T>(x2, ip, mask_zero);

		/* x = mask_zero ? x1 : (x - x2) */
		auto y = blendv<T>(sub<T>(x, x2), x1, mask_zero);
		y = blendv<T>(y, fract_x, fract_mask);
		y = blendv<T>(y, int_x, int_mask);

#ifdef DPM_PROPAGATE_NAN
		y = blendv<T>(y, x, isunord(x, x));
#endif
		return y;
	}

	__m128 modf(__m128 x, __m128 &ip) noexcept { return impl_modf<float>(x, ip); }
	__m128d modf(__m128d x, __m128d &ip) noexcept { return impl_modf<double>(x, ip); }

	template<typename T, typename I, typename V, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE Vi eval_ilogb(V abs_x) noexcept
	{
		constexpr auto do_clz = [](Vi x) noexcept
		{
			/* Mask top bits to prevent incorrect rounding. */
			x = bit_andnot(bit_shiftr<I, exp_bits<I>>(x), x);
			/* log2(x) via floating-point conversion. */
			x = bit_shiftr<I, mant_bits<I>>(std::bit_cast<Vi>(cvt<T, I>(x)));
			/* Apply exponent bias to get log2(x) using unsigned saturation. */
			constexpr I bias = std::same_as<T, double> ? 1086 : 158;
			return subs<std::uint16_t>(fill<Vi>(bias), x);
		};
		const auto ix = std::bit_cast<Vi>(abs_x);
		auto exp = bit_shiftr<I, mant_bits<I>>(ix);

		/* POSIX requires denormal numbers to be treated as if they were normalized.
		 * Shift denormal exponent by clz(ix) - (exp_bits + 1) */
		const auto fix_denorm = cmp_eq<I>(exp, setzero<Vi>());
		const auto norm_off = sub<I>(do_clz(ix), fill<Vi>(exp_bits<I> + 1));
		exp = sub<I>(exp, bit_and(norm_off, fix_denorm));

		/* Apply exponent offset. */
		return sub<I>(exp, fill<Vi>(exp_off<I>));
	}
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_ilogb(V x) noexcept
	{
		const auto abs_x = abs<T>(x);
		auto y = eval_ilogb<T, I>(abs_x);
#ifdef DPM_HANDLE_ERRORS
		y = blendv<I>(y, fill<decltype(y)>(static_cast<I>(FP_ILOGB0)), std::bit_cast<decltype(y)>(cmp_eq<T>(abs_x, setzero<V>())));
		y = blendv<I>(y, fill<decltype(y)>(std::numeric_limits<I>::max()), std::bit_cast<decltype(y)>(isinf_abs(abs_x)));
#endif
#ifdef DPM_PROPAGATE_NAN
		y = blendv<I>(y, fill<decltype(y)>(static_cast<I>(FP_ILOGBNAN)), std::bit_cast<decltype(y)>(isunord(x, x)));
#endif
		return y;
	}

	__m128i ilogb(__m128 x) noexcept { return impl_ilogb<float>(x); }
	__m128i ilogb(__m128d x) noexcept { return impl_ilogb<double>(x); }
#ifdef DPM_HAS_AVX
	__m256i ilogb(__m256 x) noexcept { return impl_ilogb<float>(x); }
	__m256i ilogb(__m256d x) noexcept { return impl_ilogb<double>(x); }
#endif

#ifndef DPM_USE_SVML
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_logb(V x) noexcept
	{
		const auto abs_x = abs<T>(x);
		auto y = cvt<T, I>(eval_ilogb<T, I>(abs_x));
#ifdef DPM_HANDLE_ERRORS
		const auto ninf = fill<V>(-std::numeric_limits<T>::infinity());
		const auto zero_mask = cmp_eq<T>(abs_x, setzero<V>());
		if (test_mask<V>(zero_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_DIVBYZERO);
			errno = ERANGE;
		}
		y = blendv<T>(y, abs_x, isinf_abs(abs_x));
		y = blendv<T>(y, ninf, zero_mask);
#endif
#ifdef DPM_PROPAGATE_NAN
		y = blendv<T>(y, x, isunord(x, x));
#endif
		return y;
	}

	__m128 logb(__m128 x) noexcept { return impl_logb<float>(x); }
	__m128d logb(__m128d x) noexcept { return impl_logb<double>(x); }

#ifdef DPM_HAS_AVX
	__m256 logb(__m256 x) noexcept { return impl_logb<float>(x); }
	__m256d logb(__m256d x) noexcept { return impl_logb<double>(x); }
#endif
#endif

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
#ifdef DPM_HANDLE_ERRORS
		/* Raise overflow if exp == inf */
		const auto inf_mask = std::bit_cast<V>(cmp_eq<I>(exp, inf_exp));
		if (test_mask<V>(bit_andnot(nan_mask, inf_mask))) [[unlikely]]
		{
			std::feraiseexcept(FE_OVERFLOW);
			errno = ERANGE;
		}
		/* Raise underflow if exp == 0 */
		const auto uflow_mask = std::bit_cast<V>(cmp_eq<I>(exp, setzero<Vi>()));
		if (test_mask<V>(bit_andnot(eq_mask, uflow_mask))) [[unlikely]]
		{
			std::feraiseexcept(FE_UNDERFLOW);
			errno = ERANGE;
		}
#endif
#endif
		/* return a == b ? b : z */
		return blendv<T>(c, b, eq_mask);
	}

	__m128 nextafter(__m128 from, __m128 to) noexcept { return impl_nextafter<float>(from, to); }
	__m128d nextafter(__m128d from, __m128d to) noexcept { return impl_nextafter<double>(from, to); }
#ifdef DPM_HAS_AVX
	__m256 nextafter(__m256 from, __m256 to) noexcept { return impl_nextafter<float>(from, to); }
	__m256d nextafter(__m256d from, __m256d to) noexcept { return impl_nextafter<double>(from, to); }
#endif
}

#endif