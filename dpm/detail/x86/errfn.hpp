/*
 * Created by switchblade on 2023-03-02.
 */

#pragma once

#include "math_fwd.hpp"
#include "type.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_USE_SVML)

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 arf(__m128 x) noexcept { return _mm_erf_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 erfc(__m128 x) noexcept { return _mm_erfc_ps(x); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d erf(__m128d x) noexcept { return _mm_erf_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d erfc(__m128d x) noexcept { return _mm_erfc_pd(x); }
#endif
#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 erf(__m256 x) noexcept { return _mm256_erf_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 erfc(__m256 x) noexcept { return _mm256_erfc_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d erf(__m256d x) noexcept { return _mm256_erf_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d erfc(__m256d x) noexcept { return _mm256_erfc_pd(x); }
#endif
	}

	/** Calculates the error function of elements in \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> erf(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(erf(packed));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::erf(x); }, result, x);
			return result;
		}
	}
	/** Calculates the complementary error function of elements in \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> erfc(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(erfc(packed));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::erfc(x); }, result, x);
			return result;
		}
	}
}

#endif