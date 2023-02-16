/*
 * Created by switchblade on 2023-02-06.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "except.hpp"
#include "polevl.hpp"

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_tan(V x) noexcept
	{
		constexpr auto extent = sizeof(V) / sizeof(T);
		const auto sign_x = masksign<T>(x);
		auto abs_x = bit_xor(x, sign_x);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = isinf_abs(abs_x); test_mask(m))
			[[unlikely]] abs_x = except_invalid<T>(abs_x, m);
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