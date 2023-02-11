/*
 * Created by switch_blade on 2023-02-10.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "cvt.hpp"

namespace dpm::detail
{
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
		constexpr I offset = std::same_as<T, double> ? 1023 : 127;
		return sub<I>(exp, fill<Vi>(offset));
	}
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_ilogb(V x) noexcept
	{
		const auto abs_x = abs<T>(x);

#ifdef DPM_HANDLE_ERRORS
		const auto ninf = fill<V>(-std::numeric_limits<T>::infinity());
		const auto inf = fill<V>(std::numeric_limits<T>::infinity());
		const auto zero_mask = cmp_eq<T>(abs_x, setzero<V>());
		const auto inf_mask = cmp_eq<T>(abs_x, inf);
#endif
#ifdef DPM_PROPAGATE_NAN
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		const auto nan_mask = isunord(x, x);
#endif

		auto y = eval_ilogb<T, I>(abs_x);
#ifdef DPM_HANDLE_ERRORS
		//y = blendv<T>(x, ninf, zero_mask);
		//y = blendv<T>(x, inf, inf_mask);
#endif
#ifdef DPM_PROPAGATE_NAN
		//y = blendv<T>(x, nan, nan_mask);
#endif
		return y;
	}

	//__m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128i x) noexcept { return impl_ilogb<float>(x); }
	//__m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128d x) noexcept { return impl_ilogb<double>(x); }
#ifdef DPM_HAS_AVX
	//__m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256i x) noexcept { return impl_ilogb<float>(x); }
	//__m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256d x) noexcept { return impl_ilogb<double>(x); }
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