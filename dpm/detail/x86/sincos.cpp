/*
 * Created by switchblade on 2023-02-01.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	template<typename T, sincos_op Op, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE sincos_ret<V> impl_sincos(V x) noexcept
	{
		const auto sign_x = masksign<T>(x);
		auto abs_x = bit_xor(x, sign_x);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = isinf_abs(abs_x); test_mask(m))
			[[unlikely]] abs_x = except_invalid<T>(abs_x, abs_x, m);
#endif
		return eval_sincos<T, Op>(sign_x, abs_x);
	}

	sincos_ret<__m128> DPM_MATHFUNC sincos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m128 DPM_MATHFUNC sin(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x).sin; }
	__m128 DPM_MATHFUNC cos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x).cos; }

	sincos_ret<__m128d> DPM_MATHFUNC sincos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SINCOS>(x); }
	__m128d DPM_MATHFUNC sin(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x).sin; }
	__m128d DPM_MATHFUNC cos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x).cos; }

#ifdef DPM_HAS_AVX
	sincos_ret<__m256> DPM_MATHFUNC sincos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m256 DPM_MATHFUNC sin(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x).sin; }
	__m256 DPM_MATHFUNC cos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x).cos; }

	sincos_ret<__m256d> DPM_MATHFUNC sincos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SINCOS>(x); }
	__m256d DPM_MATHFUNC sin(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x).sin; }
	__m256d DPM_MATHFUNC cos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x).cos; }
#endif
}

#endif