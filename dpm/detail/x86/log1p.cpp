/*
 * Created by switchblade on 2023-02-20.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	/* log(1 + x) - log(u) ~= c / u; where u = 1 + x rounded, and c = (1 + x) - u.
	 * As such, log(1 + x) ~= log(u) + c / u. When c == 0, log(1 + x) is exact.
	 *
	 * It is important to avoid underflow in c / u. As such,
	 * when u == x + 1 is in range [sqrt(2) / 2, sqrt(2)), c == 0.
	 * Otherwise, let k = (as_int(u) + 0x95f62 << 32) >> 52 - 0x3fd;
	 * if k < 52, c == 1 - (u - x) for k >= 0 and x - (u - 1) for k < 0.
	 * Otherwise (for k >= 52), c == 0.
	 *
	 * For single-precision (32-bit) floats, k = (as_int(u) + 0x4afb0d) >> 23 - 0x7d,
	 * and k threshold is 23 instead of 52 (threshold is mantissa bits).
	 *
	 * Use eval_log to calculate log(u), as there is no need for
	 * normalization since denormal x results in underflow condition. */

	template<std::same_as<float> T, typename V, typename I = std::int32_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log1p(V x) noexcept
	{
		const auto ix = std::bit_cast<Vi>(x);
		const auto u = add<T>(x, fill<V>(1.0f));

		/* sqrt(2) / 2 <= u < sqrt(2) */
		const auto range_mask = cmp_gt<I>(fill<Vi>(0xbe95f61a), ix);
		/* k = (as_int(u) + 0x4afb0d) >> 23 - 0x7d */
		auto k = add<I>(std::bit_cast<Vi>(u), fill<Vi>(0x4afb0d));
		k = sub<I>(bit_shiftr<I, 23>(k), fill<Vi>(0x7d));

		/* c = k < 0 ? x - (u - 1) : 1 - (u - x) */
		const auto c_mask = cmp_gt<I>(setzero<Vi>(), k);
		const auto c0 = sub<T>(fill<V>(1.0f), sub<T>(u, x));
		const auto c1 = sub<T>(x, sub<T>(u, fill<V>(1.0f)));
		auto c = blendv<T>(c0, c1, std::bit_cast<V>(c_mask));

		/* c = (k >= 23 || range_mask) ? 0 : c */
		const auto exact_mask = bit_or(cmp_gt<I>(k, fill<Vi>(22)), range_mask);
		c = bit_andnot(std::bit_cast<V>(exact_mask), c);

		/* log1p(x) ~= log(u) + c / u */
		auto y = add<T>(eval_log<T>(std::bit_cast<Vi>(u)), div<T>(c, u));
		/* log(0) == 0 */
		y = bit_andnot(cmp_eq<T>(x, setzero<V>()), y);
		/* Handle negative x <= -1, NaN & inf. */
		return log1p_excepts<T>(y, x);
	}
	template<std::same_as<double> T, typename V, typename I = std::int64_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log1p(V x) noexcept
	{
		const auto ix = std::bit_cast<Vi>(x);
		const auto u = add<T>(x, fill<V>(1.0));

		/* sqrt(2) / 2 <= u < sqrt(2) */
		const auto range_mask = cmp_gt<I>(fill<Vi>(0xbfd2bec4ull), ix);
		/* k = (as_int(u) + 0x4afb0d) >> 52 - 0x3fd */
		auto k = add<I>(std::bit_cast<Vi>(u), fill<Vi>(0x95f62ull << 32));
		k = sub<I>(bit_shiftr<I, 52>(k), fill<Vi>(0x3fdull));

		/* c = k < 0 ? x - (u - 1) : 1 - (u - x) */
		const auto c_mask = cmp_gt<I>(setzero<Vi>(), k);
		const auto c0 = sub<T>(fill<V>(1.0), sub<T>(u, x));
		const auto c1 = sub<T>(x, sub<T>(u, fill<V>(1.0)));
		auto c = blendv<T>(c0, c1, std::bit_cast<V>(c_mask));

		/* c = (k >= 52 || range_mask) ? 0 : c */
		const auto exact_mask = bit_or(cmp_gt<I>(k, fill<Vi>(51ull)), range_mask);
		c = bit_andnot(std::bit_cast<V>(exact_mask), c);

		/* log1p(x) ~= log(u) + c / u */
		auto y = add<T>(eval_log<T>(std::bit_cast<Vi>(u)), div<T>(c, u));
		/* log(0) == 0 */
		y = bit_andnot(cmp_eq<T>(x, setzero<V>()), y);
		/* Handle negative x <= -1, NaN & inf. */
		return log1p_excepts<T>(y, x);
	}

	__m128 DPM_MATHFUNC log1p(__m128 x) noexcept { return impl_log1p<float>(x); }
	__m128d DPM_MATHFUNC log1p(__m128d x) noexcept { return impl_log1p<double>(x); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC log1p(__m256 x) noexcept { return impl_log1p<float>(x); }
	__m256d DPM_MATHFUNC log1p(__m256d x) noexcept { return impl_log1p<double>(x); }
#endif
}
#endif