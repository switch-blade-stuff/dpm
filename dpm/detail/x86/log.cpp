/*
 * Created by switchblade on 2023-02-10.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	/* Vectorized versions of log(float) & log(double) based on implementation from the ARM optimized routines library
	 * https://github.com/ARM-software/optimized-routines license: MIT */

	template<std::same_as<float> T, typename V, typename I = std::int32_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log(V x) noexcept
	{
		const auto ix = log_normalize<T>(x);

		/* Load invc & logc constants from the lookup table. */
		const auto tmp = sub<I>(ix, fill<Vi>(0x3f33'0000));
		auto i = bit_shiftr<I, 23 - logtab_bits_f32>(tmp);
		i = bit_and(i, fill<Vi>((1 << logtab_bits_f32) - 1));
		const auto [invc, logc] = get_invc_logc<T, V>(i);

		/* x = 2^k z; where z is in range [OFF, 2 * OFF] and exact.  */
		const auto k = bit_ashiftr<I, 23>(tmp);
		auto y = std::bit_cast<V>(sub<I>(ix, bit_and(tmp, fill<Vi>(0xff80'0000))));

		/* log(x) = log1p(z/c-1) + log(c) + k*Ln2 */
		const auto r = fmsub(y, invc, fill<V>(1.0f));
		const auto y0 = fmadd(cvt<T, I>(k), fill<V>(ln2<T>), logc);

		/* Approximate log1p(r).  */
		const auto r2 = mul<T>(r, r);
		y = fmadd(fill<V>(logcoff_f32[1]), r, fill<V>(logcoff_f32[2]));
		y = fmadd(fill<V>(logcoff_f32[0]), r2, y);
		y = fmadd(y, r2, add<T>(y0, r));

		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0f)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}
	template<std::same_as<double> T, typename V, typename I = std::int64_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log(V x) noexcept
	{
		const auto ix = log_normalize<T>(x);

		/* Load invc & logc constants from the lookup table. */
		const auto tmp = sub<I>(ix, fill<Vi>(0x3fe6'9009'0000'0000));
		auto i = bit_shiftr<I, 52 - logtab_bits_f64>(tmp);
		i = bit_and(i, fill<Vi>((1 << logtab_bits_f64) - 1));
		const auto [invc, logc] = get_invc_logc<T, V>(i);

		/* x = 2^k z; where z is in range [OFF, 2 * OFF] and exact.  */
		const auto k = bit_ashiftr<I, 52>(tmp);
		auto y = std::bit_cast<V>(sub<I>(ix, bit_and(tmp, fill<Vi>(0xfffull << 52))));

		/* log(x) = log1p(z/c-1) + log(c) + k*Ln2.  */
		const auto r = fmadd(y, invc, fill<V>(-1.0));
		/* We only care about the bottom bits anyway. */
		const auto kf = cvt_i32_f64<V>(cvt_i64_i32(k));
		const auto y0 = fmadd(kf, fill<V>(ln2<T>), logc);

		/* y = r2 * (logcoff_f64[0] + r * A1 + r2 * (logcoff_f64[2] + r * logcoff_f64[3] + r2 * logcoff_f64[4])) + hi  */
		const auto r2 = mul<T>(r, r);
		const auto p = fmadd(fill<V>(logcoff_f64[1]), r, fill<V>(logcoff_f64[0]));
		y = fmadd(fill<V>(logcoff_f64[3]), r, fill<V>(logcoff_f64[2]));
		y = fmadd(fill<V>(logcoff_f64[4]), r2, y);
		y = fmadd(fmadd(y, r2, p), r2, add<T>(y0, r));

		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}

	__m128 log(__m128 x) noexcept { return impl_log<float>(x); }
	__m128d log(__m128d x) noexcept { return impl_log<double>(x); }
#ifdef DPM_HAS_AVX
	__m256 log(__m256 x) noexcept { return impl_log<float>(x); }
	__m256d log(__m256d x) noexcept { return impl_log<double>(x); }
#endif
}

#endif