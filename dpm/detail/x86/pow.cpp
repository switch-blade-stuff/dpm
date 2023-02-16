/*
 * Created by switchblade on 2023-02-14.
 */

#include "pow.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE std::pair<Vi, Vi> prepare_hypot(V &a, V &b) noexcept
	{
		/* |a|, |b| */
		auto ia = std::bit_cast<Vi>(bit_andnot(fill<V>(sign_bit<T>), a));
		auto ib = std::bit_cast<Vi>(bit_andnot(fill<V>(sign_bit<T>), b));

		/* |a| >= |b| */
		const auto ta = ia;
		const auto swap_mask = cmp_gt<I>(ib, ia);
		ia = blendv<I>(ia, ib, swap_mask); /* ia = (ia < ib) ? ib : ia */
		ib = blendv<I>(ib, ta, swap_mask); /* ib = (ia < ib) ? ia : ib */

		/* Handle domain & overflow. */
		a = std::bit_cast<V>(ia);
		b = std::bit_cast<V>(ib);
		return {ia, ib};
	}

#ifdef DPM_HAS_AVX
	[[nodiscard]] DPM_FORCEINLINE __m256 eval_hypotf(__m256 x, __m256 y, __m256 z) noexcept
	{
		/* Convert to double to prevent overflow. */
		const auto xh = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));
		const auto yh = _mm256_cvtps_pd(_mm256_extractf128_ps(y, 1));
		const auto xl = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
		const auto yl = _mm256_cvtps_pd(_mm256_castps256_ps128(y));
		const auto xyh = _mm256_cvtpd_ps(_mm256_sqrt_pd(fmadd(xh, xh, mul<double>(yh, yh))));
		const auto xyl = _mm256_cvtpd_ps(_mm256_sqrt_pd(fmadd(xl, xl, mul<double>(yl, yl))));
		return mul<float>(z, _mm256_set_m128(xyh, xyl));
	}
	[[nodiscard]] DPM_FORCEINLINE __m128 eval_hypotf(__m128 x, __m128 y, __m128 z) noexcept
	{
		/* Convert to double to prevent overflow. */
		const auto x64 = _mm256_cvtps_pd(x);
		const auto y64 = _mm256_cvtps_pd(y);
		const auto xy = fmadd(x64, x64, mul<double>(y64, y64));
		return mul<float>(z, _mm256_cvtpd_ps(_mm256_sqrt_pd(xy)));
	}
#else
	[[nodiscard]] DPM_FORCEINLINE __m128 eval_hypotf(__m128 x, __m128 y, __m128 z) noexcept
	{
		/* Convert to double to prevent overflow. */
		const auto xh = _mm_cvtps_pd(_mm_shuffle_ps(x, x, _MM_SHUFFLE(3, 2, 3, 2)));
		const auto yh = _mm_cvtps_pd(_mm_shuffle_ps(y, y, _MM_SHUFFLE(3, 2, 3, 2)));
		const auto xl = _mm_cvtps_pd(x);
		const auto yl = _mm_cvtps_pd(y);
		const auto xyh = _mm_cvtpd_ps(fmadd(xh, xh, mul<double>(yh, yh)));
		const auto xyl = _mm_cvtpd_ps(fmadd(xl, xl, mul<double>(yl, yl)));
		const auto xy = _mm_shuffle_ps(xyl, xyh, _MM_SHUFFLE(1, 0, 1, 0));
		return mul<float>(z, _mm_sqrt_ps(xy));
	}
#endif

	template<std::same_as<double> T, typename V, typename Vi = select_vector_t<std::uint64_t, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_hypot(V a, V b) noexcept
	{
		constexpr auto split = [](V x) -> std::pair<V, V>
		{
			constexpr double split_coff = 0x1p27 + 1.0;
			const auto xc = mul<T>(x, fill<V>(split_coff));
			const auto xh = add<T>(sub<T>(x, xc), xc);
			const auto xl = sub<T>(x, xh);
			const auto h = mul<T>(x, x);
			auto l = fmsub(xh, xh, h);
			l = fmadd(xh, mul<T>(xl, fill<V>(2.0)), l);
			l = fmadd(xl, xl, l);
			return {h, l};
		};

		/* Ensure that |a| >= |b| */
		const auto [ia, ib] = prepare_hypot<T>(a, b);

		/* Apply offset to avoid under- and overflow. */
		auto c = fill<V>(one<T>);
		const auto big_mask = std::bit_cast<V>(cmp_gt<std::int64_t>(ia, fill<Vi>((0x5fdull << 52) - 1)));
		const auto small_mask = std::bit_cast<V>(cmp_gt<std::int64_t>(fill<Vi>(0x23dull << 52), ib));
		const auto new_c = blendv<T>(fill<V>(0x1p-700), fill<V>(0x1p700), small_mask);
		auto ab_mult = blendv<T>(fill<V>(0x1p700), fill<V>(0x1p-700), small_mask);
		const auto off_mask = bit_or(big_mask, small_mask);
		ab_mult = blendv<T>(c, ab_mult, off_mask);
		c = blendv<T>(c, new_c, off_mask);
		/* Evaluate result without overflow. */
		const auto [hx, lx] = split(mul<T>(a, ab_mult));
		const auto [hy, ly] = split(mul<T>(b, ab_mult));
		c = mul<T>(c, sqrt(add<T>(add<T>(ly, lx), add<T>(hy, hx))));

		/* hypot(a, b) ~= a + b * b / a / 2 for small a & b */
		constexpr std::int64_t hypot_small = (64ull << 52) - 1;
		const auto ab_small = std::bit_cast<V>(cmp_gt<std::int64_t>(sub<std::uint64_t>(ia, ib), fill<Vi>(hypot_small)));

#ifdef DPM_PROPAGATE_NAN
		/* hypot(inf, NaN) = inf */
		constexpr std::uint64_t inf_mant = 0x7ffull << 52;
		const auto inf_a = std::bit_cast<V>(cmp_gt<std::int64_t>(ia, fill<Vi>(inf_mant - 1)));
		const auto inf_b = std::bit_cast<V>(cmp_eq<std::int64_t>(ib, fill<Vi>(inf_mant)));

		/* plus_mask = isinf(a) || ia - ib > small */
		const auto plus_mask = bit_andnot(inf_b, bit_or(inf_a, ab_small));
		/* return (inf_b || plus_mask) ? b + (plus_mask ? a : 0) : c */
		return blendv<T>(c, add<T>(b, bit_and(plus_mask, a)), bit_or(inf_b, plus_mask));
#else
		/* return ab_small ? a + b : c */
		return blendv<T>(c, add<T>(a, b), ab_small);
#endif
	}
	template<std::same_as<float> T, typename V, typename Vi = select_vector_t<std::uint32_t, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_hypot(V a, V b) noexcept
	{
		/* Ensure that |a| >= |b| */
		const auto [ia, ib] = prepare_hypot<T>(a, b);

		/* Apply offset to avoid under- and overflow. */
		auto c = fill<V>(one<T>);
		const auto big_mask = std::bit_cast<V>(cmp_gt<std::int32_t>(ia, fill<Vi>((0xbb << 23) - 1)));
		const auto small_mask = std::bit_cast<V>(cmp_gt<std::int32_t>(fill<Vi>(0x43 << 23), ib));
		const auto new_z = blendv<T>(fill<V>(0x1p-90f), fill<V>(0x1p90f), small_mask);
		auto ab_mult = blendv<T>(fill<V>(0x1p90f), fill<V>(0x1p-90f), small_mask);
		const auto off_mask = bit_or(big_mask, small_mask);
		ab_mult = blendv<T>(c, ab_mult, off_mask);
		c = blendv<T>(c, new_z, off_mask);

		/* Evaluate result without overflow. */
		c = eval_hypotf(mul<T>(a, ab_mult), mul<T>(b, ab_mult), c);

		/* hypot(a, b) ~= a + b for small a & b */
		constexpr std::uint32_t hypot_small = (25 << 23) - 1;
		const auto ab_small = std::bit_cast<V>(cmp_gt<std::int32_t>(sub<std::uint32_t>(ia, ib), fill<Vi>(hypot_small)));

#ifdef DPM_PROPAGATE_NAN
		/* hypot(inf, NaN) = inf */
		constexpr std::uint32_t inf_mant = 0xff << 23;
		const auto inf_a = std::bit_cast<V>(cmp_gt<std::int32_t>(ia, fill<Vi>(inf_mant - 1)));
		const auto inf_b = std::bit_cast<V>(cmp_eq<std::int32_t>(ib, fill<Vi>(inf_mant)));

		/* plus_mask = isinf(a) || ia - ib > small */
		const auto plus_mask = bit_andnot(inf_b, bit_or(inf_a, ab_small));
		/* return (inf_b || plus_mask) ? b + (plus_mask ? a : 0) : c */
		return blendv<T>(c, add<T>(b, bit_and(plus_mask, a)), bit_or(inf_b, plus_mask));
#else
		/* return ab_small ? a + b : c */
		return blendv<T>(c, add<T>(a, b), ab_small);
#endif
	}

	__m128 DPM_MATHFUNC hypot(__m128 a, __m128 b) noexcept { return impl_hypot<float>(a, b); }
	__m128d DPM_MATHFUNC hypot(__m128d a, __m128d b) noexcept { return impl_hypot<double>(a, b); }

#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC hypot(__m256 a, __m256 b) noexcept { return impl_hypot<float>(a, b); }
	__m256d DPM_MATHFUNC hypot(__m256d a, __m256d b) noexcept { return impl_hypot<double>(a, b); }
#endif
}

#endif