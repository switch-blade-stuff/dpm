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
		auto abs_x = bit_xor(x, sign_x);

		/* Check domain. */
#ifdef DPM_HANDLE_ERRORS
		const auto dom_mask = isinf_abs(abs_x);
		if (test_mask<V>(dom_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		abs_x = blendv<T>(abs_x, nan, dom_mask);
#endif

		auto p1 = undefined<V>(), p2 = undefined<V>();
		const auto p_mask = cmp_gt<T>(fill<V>(pio32<T>), abs_x);
		const auto p_mask_bits = movemask<T>(p_mask);
		if (p_mask_bits != fill_bits<extent>()) [[likely]] /* |x| >= Pi / 32 */
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
		return blendv<T>(p1, p2, p_mask);
	}

	[[nodiscard]] __m128 tan(__m128 x) noexcept { return impl_tan<float>(x); }
	[[nodiscard]] __m128d tan(__m128d x) noexcept { return impl_tan<double>(x); }

#ifdef DPM_HAS_AVX
	[[nodiscard]] __m256 tan(__m256 x) noexcept { return impl_tan<float>(x); }
	[[nodiscard]] __m256d tan(__m256d x) noexcept { return impl_tan<double>(x); }
#endif
}

#endif