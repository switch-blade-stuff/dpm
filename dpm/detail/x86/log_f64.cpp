/*
 * Created by switchblade on 2023-02-10.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "except.hpp"

namespace dpm::detail
{
	template<std::same_as<double> T, exp_op Op, typename V, typename I = std::int64_t, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_log(V x) noexcept
	{
		const auto sign_mask = fill<Vi>(1ull << 63);

		auto ix = std::bit_cast<Vi>(x);
		const auto x_sign = bit_and(ix, sign_mask);
		const auto abs_x = bit_xor(ix, x_sign);
		auto k = setzero<Vi>();

		/* Reduce x to [sqrt(2) / 2, sqrt(2)] */
		V y, f, c = setzero<V>();
		if constexpr (Op == exp_op::OP_LOG1P)
		{
			/* TODO: Handle log1p */

		}
		else
		{
			/* Normalize x. Signed comparison takes care of x_sign != 0. */
			const auto unorm_mask = std::bit_cast<Vi>(cmp_gt<T>(fill<Vi>(0x0010ull << 48), ix));
			const auto nx = std::bit_cast<Vi>(mul<T>(x, fill<V>(0x1p54)));
			k = sub<I>(k, bit_and(unorm_mask, fill<Vi>(I{54})));
			ix = blendv<I>(ix, nx, unorm_mask);
			auto hx = ix;

			const auto hx_off = fill<Vi>(0x3fe6a09eull << 32);
			hx = add<I>(hx, sub<I>(fill<Vi>(0x3ffull << 52), hx_off));
			const auto exp = bit_shiftr<I, 52>(hx);
			hx = add<I>(bit_and(hx, fill<Vi>(0xfffffull << 32)), hx_off);
			ix = bit_or(hx, bit_and(ix, fill<Vi>(0xffffffffull)));
			k = sub<I>(add<I>(k, exp), fill<Vi>(0x3ffull));
			y = std::bit_cast<V>(ix);
			f = sub<T>(y, fill<V>(1.0));
		}

		const auto hfsq = mul<T>(mul<T>(f, f), fill<V>(0.5));
		const auto s = div<T>(f, add<T>(fill<V>(2.0), f));
		const auto z = mul<T>(s, s);
		auto w = mul<T>(z, z);
		auto t1 = fmadd(w, fill<V>(logcoff_64[5]), fill<V>(logcoff_64[3]));
		t1 = fmadd(t1, w, fill<V>(logcoff_64[1]));
		t1 = mul<T>(w, t1);
		auto t2 = fmadd(w, fill<V>(logcoff_64[6]), fill<V>(logcoff_64[4]));
		t2 = fmadd(t2, w, fill<V>(logcoff_64[2]));
		t2 = fmadd(t2, w, fill<V>(logcoff_64[0]));
		const auto r = fmadd(z, t2, t1);

		/* Dispatch the selected operation. */
		if constexpr (Op == exp_op::OP_LOG || Op == exp_op::OP_LOG1P)
		{
			const auto v_ln2h = fill<V>(ln2h_64);
			const auto v_ln2l = fill<V>(ln2l_64);

			const auto kf = cvt_i32_f64<V>(cvt_i64_i32(k));
			y = fmadd(kf, v_ln2l, f);
			y = fmadd(kf, v_ln2h, add<T>(y, c));
			y = fmadd(s, add<T>(hfsq, r), y);
			y = sub<T>(y, hfsq);
		}
		else
		{
			/* log(1 + f) ~ yh + yl = f - hfsq + s * (hfsq + r) */
			auto yh = sub<T>(f, hfsq);
			ix = std::bit_cast<Vi>(yh);
			ix = bit_and(ix, fill<Vi>(0xffff'ffffull << 32));
			yh = std::bit_cast<V>(ix);
			auto yl = add<T>(hfsq, r);
			yl = fmadd(s, yl, hfsq);
			yl = sub<T>(sub<T>(f, yh), yl);

			if constexpr (Op == exp_op::OP_LOG2)
			{
				const auto v_ivln2h = fill<V>(ivln2h_64);
				const auto v_ivln2l = fill<V>(ivln2l_64);

				yl = fmadd(yl, v_ivln2h, mul<T>(add<T>(yl, yh), v_ivln2l));
				yh = mul<T>(yh, v_ivln2h);
				y = cvt_i32_f64<V>(cvt_i64_i32(k));
			}
			if constexpr (Op == exp_op::OP_LOG10)
			{
				const auto v_log10_2h = fill<V>(log10_2h_64);
				const auto v_log10_2l = fill<V>(log10_2l_64);
				const auto v_ivln10h = fill<V>(ivln10h_64);
				const auto v_ivln10l = fill<V>(ivln10l_64);

				/* log10(1 + f) + k * log10(2) ~ yh + yl */
				const auto kf = cvt_i32_f64<V>(cvt_i64_i32(k));
				y = add<T>(yl, yh);
				yl = fmadd(y, v_ivln10l, mul<T>(yl, v_ivln10h));
				yl = fmadd(kf, v_log10_2l, yl);
				yh = mul<T>(yh, v_ivln10h);
				y = mul<T>(kf, v_log10_2h);
			}

			w = add<T>(y, yh);
			y = add<T>(yl, add<T>(sub<T>(y, w), add<T>(yh, w)));
		}

		/* log(1) = 0; checking for |x| == 1, as -1 will be handled later. */
		const auto one_mask = cmp_eq<T>(std::bit_cast<V>(abs_x), fill<V>(1.0));
		y = sub<T>(y, bit_and(one_mask, y));

		/* Check for NaN (also handles infinities). */
#ifdef DPM_PROPAGATE_NAN
		/* log(inf|NaN) = inf|NaN */
		const auto max_fin = fill<Vi>((0x7ffull << 52) - 1);
		const auto nfin_mask = std::bit_cast<V>(cmp_gt<I>(abs_x, max_fin));
		y = blendv<T>(y, x, nfin_mask);
#endif

		/* Check for error conditions. */
#ifdef DPM_HANDLE_ERRORS
		if constexpr (Op == exp_op::OP_LOG1P)
		{
			/* TODO: Handle log1p */

		}
		else
		{
			/* log(-x) = NaN; log(+-0) = -inf */
			const auto zero_mask = std::bit_cast<V>(cmp_eq<I>(abs_x, setzero<Vi>()));
			auto neg_mask = std::bit_cast<V>(cmp_eq<I>(x_sign, sign_mask));
			if (test_mask(zero_mask)) [[unlikely]]
			{
				/* Only check for log(-x) if log(-0) has not been reported. */
				neg_mask = bit_andnot(zero_mask, neg_mask);
				y = except_divzero<T, -1>(y, zero_mask);
			}
			if (test_mask(neg_mask)) [[unlikely]] y = except_invalid<T>(y, neg_mask);
		}
#endif
		return y;
	}

	__m128d log(__m128d x) noexcept { return impl_log<double, exp_op::OP_LOG>(x); }
	__m128d log2(__m128d x) noexcept { return impl_log<double, exp_op::OP_LOG2>(x); }
	__m128d log10(__m128d x) noexcept { return impl_log<double, exp_op::OP_LOG10>(x); }
	//__m128d log1p(__m128d x) noexcept { return impl_log<double, exp_op::OP_LOG1P>(x); }
#ifdef DPM_HAS_AVX
	__m256d log(__m256d x) noexcept { return impl_log<double, exp_op::OP_LOG>(x); }
	__m256d log2(__m256d x) noexcept { return impl_log<double, exp_op::OP_LOG2>(x); }
	__m256d log10(__m256d x) noexcept { return impl_log<double, exp_op::OP_LOG10>(x); }
	//__m256d log1p(__m256d x) noexcept { return impl_log<double, exp_op::OP_LOG1P>(x); }
#endif
}

#endif