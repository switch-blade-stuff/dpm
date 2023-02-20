/*
 * Created by switchblade on 2023-02-20.
 */

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_frexp(V x, Vi &out_exp)
	{
		constexpr auto exp_ones = exp_mask<I> << mant_bits<I>;
		auto ix = std::bit_cast<Vi>(x);
		const auto exp_x = bit_and(ix, fill<Vi>(exp_ones));

		const auto not_fin = cmp_eq<I>(exp_x, fill<Vi>(exp_ones));
		const auto is_subn = cmp_eq<I>(exp_x, setzero<Vi>());
		const auto is_zero = cmp_eq<T>(x, setzero<V>());

		/* off = is_subn ? mant_bits + 2 : 0 */
		const auto off = bit_and(is_subn, fill<Vi>(mant_bits<I> + 2));
		auto norm_x = mul<T>(x, fill<V>(exp_mult<T>));
		const auto norm_ix = std::bit_cast<Vi>(norm_x);
		norm_x = blendv<T>(x, norm_x, std::bit_cast<V>(is_subn));
		auto norm_exp = bit_and(std::bit_cast<Vi>(norm_x), fill<Vi>(exp_ones));
		norm_exp = blendv<I>(exp_x, norm_exp, is_subn);
		ix = blendv<I>(ix, norm_ix, is_subn);

		/* norm_x = (ix & ~exp_bits) | exp_middle */
		norm_x = std::bit_cast<V>(bit_or(bit_and(ix, fill<Vi>(~exp_ones)), fill<Vi>(exp_middle<I> << mant_bits<I>)));
		norm_exp = sub<I>(sub<I>(bit_shiftr<I, mant_bits<I>>(norm_exp), fill<Vi>(exp_off<I> - 1)), off);

		const auto not_fin_or_zero = bit_or(std::bit_cast<Vi>(is_zero), not_fin);
		norm_x = blendv<T>(norm_x, add<T>(norm_x, norm_x), std::bit_cast<V>(not_fin_or_zero));
		out_exp = blendv<I>(norm_exp, exp_x, not_fin_or_zero);
		return norm_x;
	}

	__m128 DPM_MATHFUNC frexp(__m128 x, __m128i &out_exp) noexcept { return impl_frexp<float>(x, out_exp); }
	__m128d DPM_MATHFUNC frexp(__m128d x, __m128i &out_exp) noexcept { return impl_frexp<double>(x, out_exp); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC frexp(__m256 x, __m256i &out_exp) noexcept { return impl_frexp<float>(x, out_exp); }
	__m256d DPM_MATHFUNC frexp(__m256d x, __m256i &out_exp) noexcept { return impl_frexp<double>(x, out_exp); }
#endif
}

#endif