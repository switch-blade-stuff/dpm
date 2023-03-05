/*
 * Created by switchblade on 2023-02-20.
 */

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE std::pair<V, V> impl_modf(V x) noexcept
	{
		const auto ix = std::bit_cast<Vi>(x);

		/* e = ((ix >> mant_bits) & exp_mask) - exp_off */
		auto e = bit_shiftr<I, mant_bits<I>>(ix);
		e = bit_and(e, fill<Vi>(exp_mask<I>));
		e = sub<I>(e, fill<Vi>(exp_off<I>));

		/* e >= mant_bits */
		const auto int_mask = std::bit_cast<V>(cmp_gt<I>(e, fill<Vi>(mant_bits<I> - 1)));
		const auto int_x = bit_and(x, fill<V>(sign_bit<T>));

		/* e < 0 */
		const auto fract_mask = std::bit_cast<V>(cmp_gt<I>(setzero<Vi>(), e));
		const auto fract_ip = bit_and(x, fill<V>(sign_bit<T>));
		const auto fract_x = x;

		/* mask = -1 >> (exp_bits + 1) >> e */
		const auto mask = std::bit_cast<V>(bit_shiftr<I>(bit_shiftr<I, exp_bits<I> + 1>(fill<Vi>(I{-1})), e));

		/* mask_zero = (ix & mask) == 0 */
		auto mask_zero = cmp_eq<T>(bit_and(std::bit_cast<V>(ix), mask), setzero<V>());
		/* mask_zero = mask_zero && !(fract_mask | int_mask) */
		mask_zero = bit_andnot(fract_mask, mask_zero);
		mask_zero = bit_andnot(int_mask, mask_zero);

		const auto x1 = bit_and(x, fill<V>(sign_bit<T>));   /* x1 = x & -0.0 */
		const auto x2 = bit_andnot(mask, x);                /* x2 = x & ~mask */

		/* i = e < 0 ? x : x & -0.0 */
		auto i = blendv<T>(x, fract_ip, fract_mask);
		/* i = ix & mask ? x2 : i */
		i = blendv<T>(x2, i, mask_zero);

		/* x = mask_zero ? x1 : (x - x2) */
		auto y = blendv<T>(sub<T>(x, x2), x1, mask_zero);
		y = blendv<T>(y, fract_x, fract_mask);
		y = blendv<T>(y, int_x, int_mask);

#ifdef DPM_PROPAGATE_NAN
		const auto nan_mask = isunord(x, x);
		y = blendv<T>(y, x, nan_mask);
		i = blendv<T>(i, x, nan_mask);
#endif
		return {y, i};
	}

	vec2_return_t<__m128, __m128> DPM_MATHFUNC modf_f32x4(__m128 x) noexcept
	{
		const auto [y, i] = impl_modf<float>(x);
		return vec2_return(y, i);
	}
	vec2_return_t<__m128d, __m128d> DPM_MATHFUNC modf_f64x2(__m128d x) noexcept
	{
		const auto [y, i] = impl_modf<double>(x);
		return vec2_return(y, i);
	}
#ifdef DPM_HAS_AVX
	vec2_return_t<__m256, __m256i> DPM_MATHFUNC modf_f32x8(__m256 x) noexcept
	{
		const auto [y, i] = impl_modf<float>(x);
		return vec2_return(y, i);
	}
	vec2_return_t<__m256d, __m256i> DPM_MATHFUNC modf_f64x4(__m256d x) noexcept
	{
		const auto [y, i] = impl_modf<double>(x);
		return vec2_return(y, i);
	}
#endif
}

#endif