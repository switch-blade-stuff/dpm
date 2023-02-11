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
	[[nodiscard]] DPM_FORCEINLINE auto impl_modf(V x, V &ip) noexcept
	{
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		const auto ix = std::bit_cast<Vi>(x);
#ifdef DPM_PROPAGATE_NAN
		const auto nan_mask = isunord(x, x);
#endif

		/* e = ((ix >> mant_bits) & exp_mask) - exp_off */
		auto e = bit_shiftr<I, mant_bits<T>>(ix);
		e = bit_and(e, fill<Vi>(exp_mask<T>));
		e = sub<I>(e, fill<Vi>(exp_off<T>));

		/* e >= mant_bits */
		const auto int_mask = std::bit_cast<V>(cmp_gt<I>(e, fill<Vi>(mant_bits<T> - 1)));
		const auto int_x = bit_and(x, fill<V>(sign_bit<T>));

		/* e < 0 */
		const auto fract_mask = std::bit_cast<V>(cmp_gt<I>(setzero<Vi>(), e));
		const auto fract_ip = bit_and(x, fill<V>(sign_bit<T>));
		const auto fract_x = x;

		/* mask = -1 >> (exp_bits + 1) >> e */
		const auto mask = std::bit_cast<V>(bit_shiftr<I>(bit_shiftr<I, exp_bits<T> + 1>(fill<V>(I{-1})), e));

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
		x = blendv<T>(sub<T>(x, x2), x1, mask_zero);
		x = blendv<T>(x, fract_x, fract_mask);
		x = blendv<T>(x, int_x, int_mask);

#ifdef DPM_PROPAGATE_NAN
		x = blendv<T>(x, nan, nan_mask);
#endif
		return x;
	}

	__m128 DPM_API_PUBLIC DPM_MATHFUNC modf(__m128 x, __m128 &ip) noexcept { return impl_modf<float>(x, ip); }
	__m128d DPM_API_PUBLIC DPM_MATHFUNC modf(__m128d x, __m128d &ip) noexcept { return impl_modf<double>(x, ip); }

	template<typename T, typename I, typename V, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE Vi eval_ilogb(V abs_x) noexcept
	{
		constexpr auto do_clz = [](Vi x) noexcept
		{
			/* Mask top bits to prevent incorrect rounding. */
			x = bit_andnot(bit_shiftr<I, exp_bits<T>>(x), x);
			/* log2(x) via floating-point conversion. */
			x = bit_shiftr<I, mant_bits<T>>(std::bit_cast<Vi>(cvt<T, I>(x)));
			/* Apply exponent bias to get log2(x) using unsigned saturation. */
			constexpr I bias = std::same_as<T, double> ? 1086 : 158;
			return subs<std::uint16_t>(fill<Vi>(bias), x);
		};
		const auto ix = std::bit_cast<Vi>(abs_x);
		auto exp = bit_shiftr<I, mant_bits<T>>(ix);

		/* POSIX requires denormal numbers to be treated as if they were normalized.
		 * Shift denormal exponent by clz(ix) - (exp_bits + 1) */
		const auto fix_denorm = cmp_eq<I>(exp, setzero<Vi>());
		const auto norm_off = sub<I>(do_clz(ix), fill<Vi>(I{exp_bits<T> + 1}));
		exp = sub<I>(exp, bit_and(norm_off, fix_denorm));

		/* Apply exponent offset. */
		return sub<I>(exp, fill<Vi>(exp_off<T>));
	}
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_ilogb(V x) noexcept
	{
		const auto abs_x = abs<T>(x);

#ifdef DPM_HANDLE_ERRORS
		const auto zero_mask = cmp_eq<T>(abs_x, setzero<V>());
		const auto inf_mask = isinf_abs(abs_x);
#endif
#ifdef DPM_PROPAGATE_NAN
		const auto nan_mask = isunord(x, x);
#endif

		auto y = eval_ilogb<T, I>(abs_x);
#ifdef DPM_HANDLE_ERRORS
		y = blendv<I>(y, fill<decltype(y)>(std::numeric_limits<I>::max()), std::bit_cast<decltype(y)>(inf_mask));
		y = blendv<I>(y, fill<decltype(y)>(static_cast<I>(FP_ILOGB0)), std::bit_cast<decltype(y)>(zero_mask));
#endif
#ifdef DPM_PROPAGATE_NAN
		y = blendv<I>(y, fill<decltype(y)>(static_cast<I>(FP_ILOGBNAN)), std::bit_cast<decltype(y)>(nan_mask));
#endif
		return y;
	}

	__m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128 x) noexcept { return impl_ilogb<float>(x); }
	__m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128d x) noexcept { return impl_ilogb<double>(x); }
#ifdef DPM_HAS_AVX
	__m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256 x) noexcept { return impl_ilogb<float>(x); }
	__m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256d x) noexcept { return impl_ilogb<double>(x); }
#endif

#ifndef DPM_USE_SVML
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_logb(V x) noexcept
	{
		const auto abs_x = abs<T>(x);

#ifdef DPM_HANDLE_ERRORS
		const auto ninf = fill<V>(-std::numeric_limits<T>::infinity());
		const auto inf = fill<V>(std::numeric_limits<T>::infinity());
		const auto zero_mask = cmp_eq<T>(abs_x, setzero<V>());
		const auto inf_mask = cmp_eq<T>(abs_x, inf);
		if (movemask<T>(zero_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_DIVBYZERO);
			errno = ERANGE;
		}
#endif
#ifdef DPM_PROPAGATE_NAN
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		const auto nan_mask = isunord(x, x);
#endif

		x = cvt<T, I>(eval_ilogb<T, I>(abs_x));
#ifdef DPM_HANDLE_ERRORS
		x = blendv<T>(x, ninf, zero_mask);
		x = blendv<T>(x, inf, inf_mask);
#endif
#ifdef DPM_PROPAGATE_NAN
		x = blendv<T>(x, nan, nan_mask);
#endif
		return x;
	}

	__m128 DPM_API_PUBLIC DPM_MATHFUNC logb(__m128 x) noexcept { return impl_logb<float>(x); }
	__m128d DPM_API_PUBLIC DPM_MATHFUNC logb(__m128d x) noexcept { return impl_logb<double>(x); }

#ifdef DPM_HAS_AVX
	__m256 DPM_API_PUBLIC DPM_MATHFUNC logb(__m256 x) noexcept { return impl_logb<float>(x); }
	__m256d DPM_API_PUBLIC DPM_MATHFUNC logb(__m256d x) noexcept { return impl_logb<double>(x); }
#endif
#endif
}

#endif