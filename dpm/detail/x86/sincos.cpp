/*
 * Created by switchblade on 2023-02-01.
 */

#if defined(__GNUC__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	template<typename T, sincos_op Op, typename V, typename I = int_of_size_t<sizeof(T)>, typename Vi = select_vector_t<I, sizeof(V)>>
	[[nodiscard]] DPM_FORCEINLINE sincos_ret<V> impl_sincos(V x) noexcept
	{
		const auto sign_x = masksign<T>(x);
		auto abs_x = bit_xor(x, sign_x);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = isinf_abs(abs_x); test_mask(m))
			[[unlikely]] abs_x = except_invalid<T>(abs_x, abs_x, m);
#endif

		/* Check for over & underflow conditions. */
		const auto oflow_mask = cvt_has_overflow<I>(abs_x);
		/* Handle exceptional cases later. */
		auto y = eval_sincos<T, Op>(sign_x, bit_andnot(oflow_mask, abs_x));

		/* Handle exceptional cases. */
		if (const auto m = movemask<I>(oflow_mask); m) [[unlikely]]
		{
			const auto special = [](T x, T &out_sin, T &out_cos)
			{
#ifdef _GNU_SOURCE
				if constexpr (Op == sincos_op::OP_SINCOS)
				{
					if constexpr (std::same_as<T, float>)
						::sincosf(x, &out_sin, &out_cos);
					else
						::sincos(x, &out_sin, &out_cos);
					return;
				}
#endif
				if constexpr (Op & sincos_op::OP_SIN) out_sin = std::sin(x);
				if constexpr (Op & sincos_op::OP_COS) out_cos = std::cos(x);
			};
			mask_invoke<T, V, T, T &, T &>(special, m, x, y.sin, y.cos);
		}
		return y;
	}

	[[nodiscard]] vec2_return_t<__m128, __m128> DPM_PUBLIC DPM_MATHFUNC sincos_f32x4(__m128 x) noexcept
	{
		const auto [s, c] = impl_sincos<float, sincos_op::OP_SINCOS>(x);
		return vec2_return(s, c);
	}
	__m128 DPM_MATHFUNC sin(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x).sin; }
	__m128 DPM_MATHFUNC cos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x).cos; }

	[[nodiscard]] vec2_return_t<__m128d, __m128d> DPM_PUBLIC DPM_MATHFUNC sincos_f64x2(__m128d x) noexcept
	{
		const auto [s, c] = impl_sincos<double, sincos_op::OP_SINCOS>(x);
		return vec2_return(s, c);
	}
	__m128d DPM_MATHFUNC sin(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x).sin; }
	__m128d DPM_MATHFUNC cos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x).cos; }

#ifdef DPM_HAS_AVX
	[[nodiscard]] vec2_return_t<__m256, __m256> DPM_PUBLIC DPM_MATHFUNC sincos_f32x8(__m256 x) noexcept
	{
		const auto [s, c] = impl_sincos<float, sincos_op::OP_SINCOS>(x);
		return vec2_return(s, c);
	}
	__m256 DPM_MATHFUNC sin(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x).sin; }
	__m256 DPM_MATHFUNC cos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x).cos; }

	[[nodiscard]] vec2_return_t<__m256d, __m256d> DPM_PUBLIC DPM_MATHFUNC sincos_f64x4(__m256d x) noexcept
	{
		const auto [s, c] = impl_sincos<double, sincos_op::OP_SINCOS>(x);
		return vec2_return(s, c);
	}
	__m256d DPM_MATHFUNC sin(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x).sin; }
	__m256d DPM_MATHFUNC cos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x).cos; }
#endif
}

#endif