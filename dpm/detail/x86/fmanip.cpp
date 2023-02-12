/*
 * Created by switch_blade on 2023-02-10.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "class.hpp"
#include "cvt.hpp"

namespace dpm::detail
{
//	inline double_v frexp(const double_v &v, int_v *e)
//	{
//		const __m128i exponentBits = Const<double>::exponentMask().dataI();
//		const __m128i exponentPart = _mm_and_si128(_mm_castpd_si128(v.data()), exponentBits);
//		*e = _mm_sub_epi32(_mm_srli_epi64(exponentPart, 52), _mm_set1_epi32(0x3fe));
//		const __m128d exponentMaximized = _mm_or_pd(v.data(), _mm_castsi128_pd(exponentBits));
//		double_v ret = _mm_and_pd(exponentMaximized, _mm_load_pd(reinterpret_cast<const double *>(&c_general::frexpMask[0])));
//		double_m zeroMask = v == double_v::Zero();
//		ret(isnan(v) || !isfinite(v) || zeroMask) = v;
//		e->setZero(zeroMask.data());
//		return ret;
//	}
//	inline float_v frexp(const float_v &v, int_v *e)
//	{
//		const __m128i exponentBits = Const<float>::exponentMask().dataI();
//		const __m128i exponentPart = _mm_and_si128(_mm_castps_si128(v.data()), exponentBits);
//		*e = _mm_sub_epi32(_mm_srli_epi32(exponentPart, 23), _mm_set1_epi32(0x7e));
//		const __m128 exponentMaximized = _mm_or_ps(v.data(), _mm_castsi128_ps(exponentBits));
//		float_v ret = _mm_and_ps(exponentMaximized, _mm_castsi128_ps(_mm_set1_epi32(0xbf7fffff)));
//		ret(isnan(v) || !isfinite(v) || v == float_v::Zero()) = v;
//		e->setZero(v == float_v::Zero());
//		return ret;
//	}

	template<typename T, typename V, typename Vi, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_ldexp(V x, Vi n) noexcept
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

#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
		const auto not_fin = std::bit_cast<V>(cmp_eq<I>(x_exp, fill<Vi>(exp_mask<I>)));
#endif
#ifdef DPM_HANDLE_ERRORS
		const auto has_overflow = bit_or(cmp_gt<I>(n, fill<Vi>(max_ldexp<I>)), cmp_gt<I>(y_exp, fill<Vi>(exp_mask<I> + mant_bits<I> + 1)));
		const auto has_underflow = bit_or(cmp_gt<I>(fill<Vi>(-max_ldexp<I>), n), cmp_gt<I>(fill<Vi>(I{1}), y_exp));
#endif

		/* Apply the new exponent & normalize. */
		const auto exp_zeros = fill<Vi>(~(exp_mask<I> << mant_bits<I>));
		y_exp = bit_andnot(is_zero, bit_shiftl<I, mant_bits<I>>(y_exp));
		auto y = std::bit_cast<V>(bit_or(bit_and(ix, exp_zeros), y_exp));
		y = blendv<T>(y, mul<T>(y, fill<V>(exp_multm<T>)), std::bit_cast<V>(is_denorm));

#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
		y = blendv<T>(y, add<T>(x, x), not_fin);
#endif
#ifdef DPM_HANDLE_ERRORS
		if (movemask<T>(has_overflow)) [[unlikely]]
		{
			const auto vhuge = fill<V>(huge<T>);
			y = blendv<T>(y, mul<T>(vhuge, copysign<T>(vhuge, x)), std::bit_cast<V>(has_underflow));
		}
		if (movemask<T>(has_underflow)) [[unlikely]]
		{
			const auto vtiny = fill<V>(tiny<T>);
			y = blendv<T>(y, mul<T>(vtiny, copysign<T>(vtiny, x)), std::bit_cast<V>(has_underflow));
		}
#endif
		return y;
	}

	__m128 DPM_API_PUBLIC DPM_MATHFUNC ldexp(__m128 x, __m128i exp) noexcept { return impl_ldexp<float>(x,exp); }
	__m128d DPM_API_PUBLIC DPM_MATHFUNC ldexp(__m128d x, __m128i exp) noexcept { return impl_ldexp<double>(x,exp); }
#ifdef DPM_HAS_AVX
	__m256 DPM_API_PUBLIC DPM_MATHFUNC ldexp(__m256 x, __m256i exp) noexcept { return impl_ldexp<float>(x,exp); }
	__m256d DPM_API_PUBLIC DPM_MATHFUNC ldexp(__m256d x, __m256i exp) noexcept { return impl_ldexp<double>(x,exp); }
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

	__m128 DPM_API_PUBLIC DPM_MATHFUNC modf(__m128 x, __m128 &ip) noexcept { return impl_modf<float>(x, ip); }
	__m128d DPM_API_PUBLIC DPM_MATHFUNC modf(__m128d x, __m128d &ip) noexcept { return impl_modf<double>(x, ip); }

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
		auto y = cvt<T, I>(eval_ilogb<T, I>(abs_x));
#ifdef DPM_HANDLE_ERRORS
		const auto ninf = fill<V>(-std::numeric_limits<T>::infinity());
		const auto zero_mask = cmp_eq<T>(abs_x, setzero<V>());
		if (movemask<T>(zero_mask)) [[unlikely]]
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

	__m128 DPM_API_PUBLIC DPM_MATHFUNC logb(__m128 x) noexcept { return impl_logb<float>(x); }
	__m128d DPM_API_PUBLIC DPM_MATHFUNC logb(__m128d x) noexcept { return impl_logb<double>(x); }

#ifdef DPM_HAS_AVX
	__m256 DPM_API_PUBLIC DPM_MATHFUNC logb(__m256 x) noexcept { return impl_logb<float>(x); }
	__m256d DPM_API_PUBLIC DPM_MATHFUNC logb(__m256d x) noexcept { return impl_logb<double>(x); }
#endif
#endif
}

#endif