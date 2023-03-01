/*
 * Created by switchblade on 2023-02-20.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	template<std::same_as<float> T, typename V, typename I = std::int32_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log10(V x) noexcept
	{
		auto ix = std::bit_cast<Vi>(x);
		auto k = setzero<Vi>();

		const auto is_subnorm = cmp_gt<I>(fill<Vi>(0x80'0000), ix);
		if (test_mask(is_subnorm)) [[unlikely]] /* Avoid overflow in x * 2^25 */
		{
			/* Normalize x. */
			const auto ix_norm = std::bit_cast<Vi>(mul<T>(x, fill<V>(0x1p25f)));
			k = sub<I>(k, bit_and(is_subnorm, fill<Vi>(25ull)));
			ix = blendv<I>(ix, ix_norm, is_subnorm);
		}

		/* k = ilogb(x) */
		k = add<I>(k, sub<I>(bit_shiftr<I, 23>(ix), fill<Vi>(exp_off<I>)));

		/* y = scalbn(x, -(k + k < 0)) */
		const auto sign_k = bit_shiftr<I, 31>(k);
		ix = bit_and(ix, fill<Vi>(0x7f'ffff));
		ix = bit_or(ix, bit_shiftl<I, 23>(sub<I>(fill<Vi>(0x7f), sign_k)));
		auto y = cvt<T, I>(add<I>(k, sign_k));

		/* y * log10_2h + (y * log10_2l + ivln10 * log(x)) */
		auto z = mul<T>(fill<V>(ivln10<T>), eval_log<T>(ix));
		z = fmadd(y, fill<V>(log10_2l<T>), z);
		y = fmadd(y, fill<V>(log10_2h<T>), z);

		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0f)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}
	template<std::same_as<double> T, typename V, typename I = std::int64_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log10(V x) noexcept
	{
		auto ix = std::bit_cast<Vi>(x);
		auto k = setzero<Vi>();

		const auto top16 = bit_shiftr<I, 48>(ix);
		const auto is_subnorm = cmp_gt<I>(fill<Vi>(0x10ull << 48), top16);
		if (test_mask(is_subnorm)) [[unlikely]] /* Avoid overflow in x * 2^54 */
		{
			/* Normalize x. */
			const auto ix_norm = std::bit_cast<Vi>(mul<T>(x, fill<V>(0x1p54)));
			k = sub<I>(k, bit_and(is_subnorm, fill<Vi>(54ull)));
			ix = blendv<I>(ix, ix_norm, is_subnorm);
		}

		/* k = ilogb(x) */
		k = add<I>(k, sub<I>(bit_shiftr<I, 52>(ix), fill<Vi>(exp_off<I>)));

		/* y = scalbn(x, -(k + k < 0)) */
		const auto sign_k = bit_shiftr<I, 63>(k);
		ix = bit_and(ix, fill<Vi>(0xf'ffff'ffff'ffff));
		ix = bit_or(ix, bit_shiftl<I, 52>(sub<I>(fill<Vi>(0x3ffull), sign_k)));
		auto y = cvt_i32_f64<V>(cvt_i64_i32(add<I>(k, sign_k)));

		/* y * log10_2h + (y * log10_2l + ivln10 * log(x)) */
		auto z = mul<T>(fill<V>(ivln10<T>), eval_log<T>(ix));
		z = fmadd(y, fill<V>(log10_2l<T>), z);
		y = fmadd(y, fill<V>(log10_2h<T>), z);

		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}

	__m128 DPM_MATHFUNC log10(__m128 x) noexcept { return impl_log10<float>(x); }
	__m128d DPM_MATHFUNC log10(__m128d x) noexcept { return impl_log10<double>(x); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC log10(__m256 x) noexcept { return impl_log10<float>(x); }
	__m256d DPM_MATHFUNC log10(__m256d x) noexcept { return impl_log10<double>(x); }
#endif
}
#endif