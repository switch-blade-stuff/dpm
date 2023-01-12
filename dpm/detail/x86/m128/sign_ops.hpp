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
#if defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH)
		[[nodiscard]] inline __m128d DPM_TARGET("sse2") x86_abs(__m128d value) noexcept { return _mm_and_pd(value, _mm_set1_pd(std::bit_cast<double>(0x7fff'ffff'ffff'ffff))); }
		[[nodiscard]] inline __m128d DPM_TARGET("sse2") x86_masksign(__m128d value) noexcept { return _mm_or_pd(x86_abs(value), _mm_set1_pd(std::bit_cast<double>(0x8000'0000'0000'0000))); }
		[[nodiscard]] inline __m128d DPM_TARGET("sse2") x86_copysign(__m128d value, __m128d mask) noexcept { return _mm_or_pd(x86_abs(value), mask); }
#endif

		[[nodiscard]] inline __m128 x86_abs(__m128 value) noexcept { return _mm_and_ps(value, _mm_set1_ps(std::bit_cast<float>(0x7fff'ffff))); }
		[[nodiscard]] inline __m128 x86_masksign(__m128 value) noexcept { return _mm_or_ps(x86_abs(value), _mm_set1_ps(std::bit_cast<float>(0x8000'0000))); }
		[[nodiscard]] inline __m128 x86_copysign(__m128 value, __m128 mask) noexcept { return _mm_or_ps(x86_abs(value), mask); }
	}

#ifdef DPM_HAS_SSE2
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fabs(const simd<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_abs(ext::to_native_data(value)[i]);
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<double, detail::avec<N, A>> signbit(const simd<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd_mask<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
		{
			const auto sign = detail::x86_masksign(ext::to_native_data(value)[i]);
			ext::to_native_data(result)[i] = _mm_cmpneq_pd(sign, _mm_setzero_pd());
		}
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> copysign(const simd<double, detail::avec<N, A>> &value, const simd<double, detail::avec<N, A>> &sign) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		simd<double, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
		{
			const auto mask = detail::x86_masksign(ext::to_native_data(sign)[i]).second;
			ext::to_native_data(result)[i] = detail::x86_copysign(ext::to_native_data(value)[i], mask);
		}
		return result;
	}
#endif

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> fabs(const simd<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
			ext::to_native_data(result)[i] = detail::x86_abs(ext::to_native_data(value)[i]);
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd_mask<float, detail::avec<N, A>> signbit(const simd<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd_mask<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
		{
			const auto sign = detail::x86_masksign(ext::to_native_data(value)[i]);
			ext::to_native_data(result)[i] = _mm_cmpneq_ps(sign, _mm_setzero_ps());
		}
		return result;
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<float, detail::avec<N, A>> copysign(const simd<float, detail::avec<N, A>> &value, const simd<float, detail::avec<N, A>> &sign) noexcept requires detail::x86_overload_m128<float, N, A>
	{
		simd<float, detail::avec<N, A>> result;
		for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
		{
			const auto mask = detail::x86_masksign(ext::to_native_data(sign)[i]).second;
			ext::to_native_data(result)[i] = detail::x86_copysign(ext::to_native_data(value)[i], mask);
		}
		return result;
	}
}

#endif