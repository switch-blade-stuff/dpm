/*
 * Created by switchblade on 2023-01-11.
 */

#pragma once

#include "type.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 abs(__m128 x) noexcept { return _mm_and_ps(x, _mm_set1_ps(std::bit_cast<float>(0x7fff'ffff))); }
		[[nodiscard]] DPM_FORCEINLINE __m128 masksign(__m128 x) noexcept { return _mm_and_ps(x, _mm_set1_ps(-0.0f)); }
		[[nodiscard]] DPM_FORCEINLINE __m128 copysign(__m128 x, __m128 m) noexcept { return _mm_or_ps(abs(x), m); }
	}

#ifdef DPM_HAS_SSE
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<float, detail::avec<N, A>> fabs(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::overload_128<float, N, A>
	{
		simd<float, detail::avec<N, A>> result = {};
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::abs(ext::to_native_data(x)[i]);
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd_mask<float, detail::avec<N, A>> signbit(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::overload_128<float, N, A>
	{
		simd_mask<float, detail::avec<N, A>> result = {};
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
		{
			const auto sign = detail::masksign(ext::to_native_data(x)[i]);
			ext::to_native_data(result)[i] = _mm_cmpneq_ps(sign, _mm_setzero_ps());
		}
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<float, detail::avec<N, A>> copysign(const simd<float, detail::avec<N, A>> &x, const simd<float, detail::avec<N, A>> &sign) noexcept requires detail::overload_128<float, N, A>
	{
		simd<float, detail::avec<N, A>> result = {};
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
		{
			const auto mask = detail::masksign(ext::to_native_data(sign)[i]);
			ext::to_native_data(result)[i] = detail::copysign(ext::to_native_data(x)[i], mask);
		}
		return result;
	}
#endif
}

#endif