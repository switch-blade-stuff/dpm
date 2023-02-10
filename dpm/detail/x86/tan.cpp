/*
 * Created by switchblade on 2023-02-06.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "polevl.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_tan(V x) noexcept
	{
		constexpr auto extent = sizeof(V) / sizeof(T);
		const auto sign_x = masksign<T>(x);
		const auto abs_x = bit_xor(x, sign_x);

#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
#ifdef DPM_HANDLE_ERRORS
		auto dom_mask = isinf_abs(abs_x);
		if (movemask<T>(dom_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
		const auto nan_mask = bit_or(dom_mask, isunord(x, x));
#else
		const auto nan_mask = isunord(x, x);
#endif
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		if (movemask<T>(nan_mask) == fill_bits<extent>()) [[unlikely]]
			return nan;
#endif

		auto p1 = undefined<V>(), p2 = undefined<V>();
		const auto p_mask = cmp_lt<T>(abs_x, fill<V>(pio32<T>));
		const auto p_mask_bits = movemask<T>(p_mask);
		if (p_mask_bits != fill_bits<extent>()) /* |x| >= Pi / 32 */
		{
			/* tan(x) = sin(x) / cos(x) */
			const auto [sin_x, cos_x] = eval_sincos(sign_x, abs_x);
			p1 = div<T>(sin_x, cos_x);
		}
		if (p_mask_bits != 0) [[unlikely]] /* |x| < Pi / 32 */
		{
			const auto xx = mul<T>(x, x);
			p2 = polevl(xx, std::span{tancof<T>});
			p2 = mul<T>(p2, x);
		}

		/* Select correct result & handle NaN. */
		auto y = blendv<T>(p1, p2, p_mask);
#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
		y = blendv<T>(y, nan, nan_mask);
#endif
		return y;
	}

	[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC tan(__m128 x) noexcept { return impl_tan<float>(x); }
	[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC tan(__m128d x) noexcept { return impl_tan<double>(x); }

#ifdef DPM_HAS_AVX
	[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC tan(__m256 x) noexcept { return impl_tan<float>(x); }
	[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC tan(__m256d x) noexcept { return impl_tan<double>(x); }
#endif
}

#endif