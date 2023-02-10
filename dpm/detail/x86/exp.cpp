/*
 * Created by switch_blade on 2023-02-10.
 */

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "cvt.hpp"

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE Vi eval_ilogb(V abs_x)
	{
		constexpr auto do_clz = [](Vi x) noexcept
		{
			/* Mask top bits to prevent incorrect rounding. */
			x = bit_andnot(bit_shiftr<I, exp_bits<T>>(x), x);
			/* log2(x) via floating-point conversion. */
			x = bit_shiftr<I, mant_bits<T>>(std::bit_cast<Vi>(cvt<T, I>(x)));
			/* Apply exponent bias to get log2(x) using unsigned saturation. */
			constexpr I bias = std::same_as<T, double> ? 1086 : 158;
			return subs<std::uint16_t>(fill<Vi>(bias), x);
		};
		const auto ix = std::bit_cast<Vi>(abs_x);
		auto exp = bit_shiftr<I, mant_bits<T>>(ix);

		/* POSIX requires denormal numbers to be treated as if they were normalized.
		 * Shift denormal exponent by clz(ix) - (exp_bits + 1) */
		const auto fix_denorm = cmp_eq<I>(exp, setzero<Vi>());
		const auto norm_off = sub<I>(do_clz(ix), fill<Vi>(I{exp_bits<T> + 1}));
		exp = sub<I>(exp, bit_and(norm_off, fix_denorm));

		/* Apply exponent offset. */
		constexpr I offset = std::same_as<T, double> ? 1023 : 127;
		return sub<I>(exp, fill<Vi>(offset));
	}

	DPM_FORCEINLINE __m128 logb(__m128 x)
	{
		const auto sign_bit = _mm_set1_ps(-0.0f);
		const auto abs_x = _mm_andnot_ps(sign_bit, x);
		const auto nan = _mm_set1_ps(std::numeric_limits<float>::quiet_NaN());
		const auto ninf = _mm_set1_ps(-std::numeric_limits<float>::infinity());
		const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
		const auto zero_mask = _mm_cmpeq_ps(abs_x, _mm_set1_ps(0));
		const auto inf_mask = _mm_cmpeq_ps(abs_x, inf);
		const auto nan_mask = _mm_cmpunord_ps(x, x);

		x = _mm_cvtepi32_ps(eval_ilogb<float>(abs_x));
		x = _mm_blendv_ps(x, ninf, zero_mask);
		x = _mm_blendv_ps(x, inf, inf_mask);
		x = _mm_blendv_ps(x, nan, nan_mask);
		return x;
	}
}
#endif