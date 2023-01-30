/*
 * Created by switchblade on 2023-01-15.
 */

#pragma once

#include "type.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm
{
	namespace detail
	{
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d fmadd_sse(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_add_pd(_mm_mul_pd(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d fmsub_sse(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_sub_pd(_mm_mul_pd(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d fnmadd_sse(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_sub_pd(c, _mm_mul_pd(a, b));
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128d fnmsub_sse(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_sub_pd(_mm_setzero_pd(), fmadd_sse(a, b, c));
		}

		[[nodiscard]] DPM_FORCEINLINE __m128d fmadd(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmadd_pd(a, b, c);
#else
			return fmadd_sse(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d fmsub(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmsub_pd(a, b, c);
#else
			return fmsub_sse(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d fnmadd(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmadd_pd(a, b, c);
#else
			return fnmadd_sse(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d fnmsub(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmsub_pd(a, b, c);
#else
			return fnmsub_sse(a, b, c);
#endif
		}
	}

#ifdef DPM_HAS_SSE2
	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> fmadd(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result = {};
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::fmadd(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> fmsub(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result = {};
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::fmsub(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> fnmadd(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result = {};
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::fnmadd(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> fnmsub(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result = {};
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::fnmsub(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> fma(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b,
			const simd<double, detail::avec<N, A>> &c)
	noexcept requires detail::x86_overload_128<double, N, A>
	{
		return ext::fmadd(a, b, c);
	}
#endif
}

#endif