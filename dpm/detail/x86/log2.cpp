/*
 * Created by switchblade on 2023-02-21.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	/* Vectorized versions of log2(float) & log2(double) based on implementation from the ARM optimized routines library
	 * https://github.com/ARM-software/optimized-routines license: MIT */
	template<std::same_as<float> T, typename V, typename I = std::int32_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log2(V x) noexcept
	{
		/* Normalize x. */
		auto ix = log_normalize<T>(x);

		/* Load invc & logc constants from the lookup table. */
		const auto tmp = sub<I>(ix, fill<Vi>(0x3f33'0000));
		auto i = bit_shiftr<I, 23 - log2tab_bits_f32>(tmp);
		i = bit_and(i, fill<Vi>((1 << log2tab_bits_f32) - 1));
		const auto [invc, logc] = log_get_table<V>(i, log2tab_v<T>, DPM_SOURCE_LOC_CURRENT);

		/* x = 2^k z; where z is in range [0x3f33'0000, 2 * 0x3f33'0000] and exact.  */
		const auto z = std::bit_cast<V>(sub<I>(ix, bit_and(tmp, fill<Vi>(0xff80'0000))));
		const auto k = bit_ashiftr<I, 23>(tmp);

		/* log2(x) = log1p(z/c-1) / ln2 + log2(c) + k */
		const auto r = fmadd(z, invc, fill<V>(-1.0f));
		const auto y0 = add<T>(logc, cvt<T, I>(k));

		/* Approximate log1p(r) / ln2. */
		const auto r2 = mul<T>(r, r);
		const auto p = fmadd(fill<V>(log2coff_f32[3]), r, y0);
		auto y = fmadd(fill<V>(log2coff_f32[1]), r, fill<V>(log2coff_f32[2]));
		y = fmadd(fmadd(fill<V>(log2coff_f32[0]), r2, y), r2, p);

		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0f)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}
	template<std::same_as<double> T, typename V, typename I = std::int64_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log2(V x) noexcept
	{
		/* Normalize x. */
		auto ix = log_normalize<T>(x);

		/* Load invc & logc constants from the lookup table. */
		const auto tmp = sub<I>(ix, fill<Vi>(0x3fe6'9009'0000'0000));
		auto i = bit_shiftr<I, 52 - log2tab_bits_f64>(tmp);
		i = bit_and(i, fill<Vi>((1 << log2tab_bits_f64) - 1));
		const auto [invc, logc] = log_get_table<V>(i, log2tab_v<T>, DPM_SOURCE_LOC_CURRENT);

		/* x = 2^k z; where z is in range [0x3fe6'9009'0000'0000, 2 * 0x3fe6'9009'0000'0000] and exact.  */
		const auto k = bit_ashiftr<I, 52>(tmp);
		const auto z = std::bit_cast<V>(sub<I>(ix, bit_and(tmp, fill<Vi>(0xfffull << 52))));

		/* log2(x) = log2(z/c) + log2(c) + k.  */
		const auto r = fmadd(z, invc, fill<V>(-1.0));
		const auto t1 = mul<T>(r, fill<V>(invln2h<T>));
		auto t2 = fmsub(r, fill<V>(invln2h<T>), t1);
		t2 = fmadd(r, fill<V>(invln2l<T>), t2);

		/* hi + lo = r / ln2 + log2(c) + k.  */
		const auto kf = cvt_i32_f64<V>(cvt_i64_i32(k));
		const auto t3 = add<T>(logc, kf);
		const auto hi = add<T>(t3, t1);
		const auto lo = add<T>(sub<T>(t3, hi), add<T>(t1, t2));

		/* log2(r + 1) = r / ln2 + r ^ 2 * poly(r).  */
		const auto r2 = mul<T>(r, r);
		const auto r4 = mul<T>(r2, r2);
		const auto p = fmadd(r, fill<V>(log2coff_f64[3]), fill<V>(log2coff_f64[2]));
		auto y = fmadd(fill<V>(log2coff_f64[5]), r, fill<V>(log2coff_f64[4]));
		y = fmadd(r4, y, fill<V>(log2coff_f64[0]));
		y = fmadd(r, fill<V>(log2coff_f64[1]), y);
		y = fmadd(r2, fmadd(r2, p, y), hi);
		y = add<T>(lo, y);

		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}

	__m128 DPM_MATHFUNC log2(__m128 x) noexcept { return impl_log2<float>(x); }
	__m128d DPM_MATHFUNC log2(__m128d x) noexcept { return impl_log2<double>(x); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC log2(__m256 x) noexcept { return impl_log2<float>(x); }
	__m256d DPM_MATHFUNC log2(__m256d x) noexcept { return impl_log2<double>(x); }
#endif
}
#endif