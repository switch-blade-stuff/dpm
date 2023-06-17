/*
 * Created by switchblade on 2023-02-20.
 */

#include <experimental/simd>

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "except.hpp"
#include "class.hpp"
#include "cvt.hpp"

namespace dpm::detail
{
	template<typename T, typename I, typename V, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE Vi eval_ilogb(V abs_x) noexcept
	{
		const auto ix = std::bit_cast<Vi>(abs_x);
		auto exp = bit_shiftr<I, mant_bits<I>>(ix);

		/* POSIX requires denormal numbers to be treated as if they were normalized.
		 * Shift denormal exponent by clz(ix) - (exp_bits + 1) */
		const auto fix_denorm = cmp_eq<I>(exp, setzero<Vi>());

		/* Mask top bits to prevent incorrect rounding. */
		auto x_clz = bit_andnot(bit_shiftr<I, exp_bits<I>>(ix), ix);
		/* log2(x) via floating-point conversion. */
		x_clz = bit_shiftr<I, mant_bits<I>>(std::bit_cast<Vi>(cvt<T, I>(x_clz)));
		/* Apply exponent bias to get log2(x) using unsigned saturation. */
		constexpr I bias = std::same_as<T, double> ? 1086 : 158;
		x_clz = subs<std::uint16_t>(fill<Vi>(bias), x_clz);

		const auto norm_off = sub<I>(x_clz, fill<Vi>(exp_bits<I> + 1));
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

	__m128i DPM_MATHFUNC ilogb(__m128 x) noexcept { return impl_ilogb<float>(x); }
	__m128i DPM_MATHFUNC ilogb(__m128d x) noexcept { return impl_ilogb<double>(x); }
#ifdef DPM_HAS_AVX
	__m256i DPM_MATHFUNC ilogb(__m256 x) noexcept { return impl_ilogb<float>(x); }
	__m256i DPM_MATHFUNC ilogb(__m256d x) noexcept { return impl_ilogb<double>(x); }
#endif

#ifndef DPM_USE_SVML
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_logb(V x) noexcept
	{
		const auto abs_x = abs<T>(x);
		auto y = cvt<T, I>(eval_ilogb<T, I>(abs_x));
#if defined(DPM_HANDLE_ERRORS)
		if (const auto m = cmp_eq<T>(abs_x, setzero<V>()); test_mask(m))
			[[unlikely]] y = except_divzero<T, -1>(y, abs_x, m);

		/* logb(+-inf) = inf; logb(+-nan) = nan */
		const auto dom_mask = bit_or(isinf_abs(abs_x), isunord(x, x));
		if (test_mask(dom_mask)) [[unlikely]] y = blendv<T>(y, mul<T>(x, x), dom_mask);
#elif defined(DPM_PROPAGATE_NAN)
		/* logb(+-nan) = nan */
		const auto dom_mask = isunord(x, x);
		if (test_mask(dom_mask)) [[unlikely]] y = blendv<T>(y, mul<T>(x, x), dom_mask);
#endif
		return y;
	}

	__m128 DPM_MATHFUNC logb(__m128 x) noexcept { return impl_logb<float>(x); }
	__m128d DPM_MATHFUNC logb(__m128d x) noexcept { return impl_logb<double>(x); }

#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC logb(__m256 x) noexcept { return impl_logb<float>(x); }
	__m256d DPM_MATHFUNC logb(__m256d x) noexcept { return impl_logb<double>(x); }
#endif
#endif
}

#endif