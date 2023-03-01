/*
 * Created by switchblade on 2023-02-20.
 */

#include "fmanip.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE V impl_nextafter(V a, V b)
	{
		auto ia = std::bit_cast<Vi>(a);
		auto ib = std::bit_cast<Vi>(b);
		const auto a_sign = std::bit_cast<Vi>(bit_and(a, fill<V>(sign_bit<T>)));
		const auto b_sign = std::bit_cast<Vi>(bit_and(b, fill<V>(sign_bit<T>)));
		const auto abs_a = bit_xor(ia, a_sign);
		const auto abs_b = bit_xor(ib, b_sign);

		const auto zero_mask = cmp_eq<I>(abs_a, setzero<Vi>());
		/* x_off = (abs_a > abs_b || (a_sign ^ b_sign)) ? -1 : 1 */
		const auto sub_mask = bit_or(cmp_gt<I>(setzero<Vi>(), bit_xor(a_sign, b_sign)), cmp_gt<I>(abs_a, abs_b));
		const auto x_off = bit_or(fill<Vi>(I{1}), sub_mask);
		/* ix = ax == 0 ? y_sign | 1 : ix + x_off */
		ia = blendv<I>(add<I>(ia, x_off), bit_or(b_sign, fill<Vi>(I{1})), zero_mask);
		auto c = std::bit_cast<V>(ia);

		/* Check domain & propagate NaN */
		const auto eq_mask = cmp_eq<T>(a, b);
#ifdef DPM_PROPAGATE_NAN
		/* c = isnan(a) || isnan(b) ? a | b : c */
		const auto nan_mask = isunord(a, b);
		c = blendv<T>(c, bit_or(a, b), nan_mask);
#endif
		/* return a == b ? b : c */
		return blendv<T>(c, b, eq_mask);
	}

	__m128 DPM_MATHFUNC nextafter(__m128 from, __m128 to) noexcept { return impl_nextafter<float>(from, to); }
	__m128d DPM_MATHFUNC nextafter(__m128d from, __m128d to) noexcept { return impl_nextafter<double>(from, to); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC nextafter(__m256 from, __m256 to) noexcept { return impl_nextafter<float>(from, to); }
	__m256d DPM_MATHFUNC nextafter(__m256d from, __m256d to) noexcept { return impl_nextafter<double>(from, to); }
#endif
}

#endif