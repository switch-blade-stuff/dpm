/*
 * Created by switchblade on 2023-02-20.
 */

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE std::pair<V, Vi> impl_frexp(V x)
	{
		constexpr auto exp_ones = exp_mask<I> << mant_bits<I>;
		auto ix = std::bit_cast<Vi>(x);
		const auto exp_x = bit_and(ix, fill<Vi>(exp_ones));

		const auto not_fin = cmp_eq<I>(exp_x, fill<Vi>(exp_ones));
		const auto is_subn = cmp_eq<I>(exp_x, setzero<Vi>());
		const auto is_zero = cmp_eq<T>(x, setzero<V>());

		/* off = is_subn ? mant_bits + 2 : 0 */
		const auto off = bit_and(is_subn, fill<Vi>(std::numeric_limits<I>::digits));
		auto norm_x = mul<T>(bit_and(std::bit_cast<V>(is_subn), x), fill<V>(exp_mult<T>));
		norm_x = blendv<T>(x, norm_x, std::bit_cast<V>(is_subn));
		auto norm_exp = bit_and(std::bit_cast<Vi>(norm_x), fill<Vi>(exp_ones));
		ix = std::bit_cast<Vi>(norm_x);

		/* norm_x = (ix & ~exp_bits) | exp_middle */
		norm_x = std::bit_cast<V>(bit_or(bit_and(ix, fill<Vi>(~exp_ones)), fill<Vi>(exp_middle<I> << mant_bits<I>)));
		norm_exp = sub<I>(sub<I>(bit_shiftr<I, mant_bits<I>>(norm_exp), fill<Vi>(exp_off<I> - 1)), off);

		const auto not_fin_or_zero = bit_or(std::bit_cast<Vi>(is_zero), not_fin);
		norm_x = blendv<T>(norm_x, x, std::bit_cast<V>(not_fin_or_zero));
		norm_exp = blendv<I>(norm_exp, exp_x, not_fin_or_zero);
		return {norm_x, norm_exp};
	}

	vec2_return_t<__m128, __m128i> DPM_MATHFUNC frexp_f32x4(__m128 x) noexcept
	{
		const auto [y, e] = impl_frexp<float>(x);
		return vec2_return(y, e);
	}
	vec2_return_t<__m128d, __m128i> DPM_MATHFUNC frexp_f64x2(__m128d x) noexcept
	{
		const auto [y, e] = impl_frexp<double>(x);
		return vec2_return(y, e);
	}
#ifdef DPM_HAS_AVX
	vec2_return_t<__m256, __m256i> DPM_MATHFUNC frexp_f32x8(__m256 x) noexcept
	{
		const auto [y, e] = impl_frexp<float>(x);
		return vec2_return(y, e);
	}
	vec2_return_t<__m256d, __m256i> DPM_MATHFUNC frexp_f64x4(__m256d x) noexcept
	{
		const auto [y, e] = impl_frexp<double>(x);
		return vec2_return(y, e);
	}
#endif
}

#endif