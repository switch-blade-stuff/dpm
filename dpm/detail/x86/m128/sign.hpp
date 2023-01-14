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
		[[nodiscard]] inline __m128d x86_abs(__m128d x) noexcept { return _mm_and_pd(x, _mm_set1_pd(std::bit_cast<double>(0x7fff'ffff'ffff'ffff))); }
		[[nodiscard]] inline __m128d x86_masksign(__m128d x) noexcept { return _mm_or_pd(x86_abs(x), _mm_set1_pd(-0.0)); }
		[[nodiscard]] inline __m128d x86_copysign(__m128d x, __m128d m) noexcept { return _mm_or_pd(x86_abs(x), m); }
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fabs(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_abs(ext::to_native_data(x)[i]);
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<double, detail::avec<N, A>> signbit(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd_mask<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
		{
			const auto sign = detail::x86_masksign(ext::to_native_data(x)[i]);
			ext::to_native_data(result)[i] = _mm_cmpneq_pd(sign, _mm_setzero_pd());
		}
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> copysign(const simd<double, detail::avec<N, A>> &x, const simd<double, detail::avec<N, A>> &sign) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
		{
			const auto mask = detail::x86_masksign(ext::to_native_data(sign)[i]).second;
			ext::to_native_data(result)[i] = detail::x86_copysign(ext::to_native_data(x)[i], mask);
		}
		return result;
	}
#endif

	namespace detail
	{
		[[nodiscard]] inline __m128 x86_abs(__m128 x) noexcept { return _mm_and_ps(x, _mm_set1_ps(std::bit_cast<float>(0x7fff'ffff))); }
		[[nodiscard]] inline __m128 x86_masksign(__m128 x) noexcept { return _mm_or_ps(x86_abs(x), _mm_set1_ps(-0.0f)); }
		[[nodiscard]] inline __m128 x86_copysign(__m128 x, __m128 m) noexcept { return _mm_or_ps(x86_abs(x), m); }
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> fabs(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_abs(ext::to_native_data(x)[i]);
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<float, detail::avec<N, A>> signbit(const simd<float, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd_mask<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
		{
			const auto sign = detail::x86_masksign(ext::to_native_data(x)[i]);
			ext::to_native_data(result)[i] = _mm_cmpneq_ps(sign, _mm_setzero_ps());
		}
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> copysign(const simd<float, detail::avec<N, A>> &x, const simd<float, detail::avec<N, A>> &sign) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
		{
			const auto mask = detail::x86_masksign(ext::to_native_data(sign)[i]).second;
			ext::to_native_data(result)[i] = detail::x86_copysign(ext::to_native_data(x)[i], mask);
		}
		return result;
	}
}

#endif