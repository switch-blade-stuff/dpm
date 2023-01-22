/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "type.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include <emmintrin.h>

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] std::pair<__m128d, __m128d> DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_sincos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_sin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_cos(__m128d x) noexcept;
	}

#ifdef DPM_HAS_SSE2
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> sin(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_sin(ext::to_native_data(x)[i]);
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> cos(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_cos(ext::to_native_data(x)[i]);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t A>
		inline void sincos(const simd<double, detail::avec<N, A>> &x, simd<double, detail::avec<N, A>> &out_sin, simd<double, detail::avec<N, A>> &out_cos) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				const auto [sin, cos] = detail::x86_sincos(ext::to_native_data(x)[i]);
				ext::to_native_data(out_sin)[i] = sin;
				ext::to_native_data(out_cos)[i] = cos;
			}
		}
	}
#endif
}

#endif