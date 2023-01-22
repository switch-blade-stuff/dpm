/*
 * Created by switchblade on 2023-01-11.
 */

#pragma once

#include "sign.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] inline DPM_FORCEINLINE __m128d x86_isnan(__m128d x) noexcept { return _mm_cmpunord_pd(x, x); }
		[[nodiscard]] inline DPM_FORCEINLINE __m128d x86_isinf(__m128d x) noexcept
		{
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
			return _mm_cmpeq_pd(x86_abs(x), inf);
		}
	}

#ifdef DPM_HAS_SSE2
	/** Determines is elements of \a x are unordered NaN and returns the resulting mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd_mask<double, detail::avec<N, A>> isnan(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd_mask<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_isnan(ext::to_native_data(x)[i]);
		return result;
	}
	/** Determines is elements of \a x are infinite and returns the resulting mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd_mask<double, detail::avec<N, A>> isinf(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd_mask<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_isinf(ext::to_native_data(x)[i]);
		return result;
	}
#endif
}

#endif