/*
 * Created by switchblade on 2023-01-11.
 */

#pragma once

#include "type.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm
{
#ifdef DPM_HAS_SSE2
	namespace detail
	{
		[[nodiscard]] inline __m128d x86_isnan(__m128d x) noexcept { return _mm_cmpunord_pd(x, x); }
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<double, detail::avec<N, A>> isnan(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd_mask<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_isnan(ext::to_native_data(x)[i]);
		return result;
	}
#endif

	namespace detail
	{
		[[nodiscard]] inline __m128 x86_isnan(__m128 x) noexcept { return _mm_cmpunord_ps(x, x); }
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<float, detail::avec<N, A>> isnan(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd_mask<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_isnan(ext::to_native_data(x)[i]);
		return result;
	}
}

#endif