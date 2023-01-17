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
		[[maybe_unused]] [[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fmadd_sse2(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_add_pd(_mm_mul_pd(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fmsub_sse2(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_sub_pd(_mm_mul_pd(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fnmadd_sse2(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_sub_pd(c, _mm_mul_pd(a, b));
		}
		[[maybe_unused]] [[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fnmsub_sse2(__m128d a, __m128d b, __m128d c) noexcept
		{
			return _mm_sub_pd(_mm_setzero_pd(), x86_fmadd_sse2(a, b, c));
		}

		[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fmadd(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmadd_pd(a, b, c);
#else
			return x86_fmadd_sse2(a, b, c);
#endif
		}
		[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fmsub(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmsub_pd(a, b, c);
#else
			return x86_fmsub_sse2(a, b, c);
#endif
		}
		[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fnmadd(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmadd_pd(a, b, c);
#else
			return x86_fnmadd_sse2(a, b, c);
#endif
		}
		[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_fnmsub(__m128d a, __m128d b, __m128d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmsub_pd(a, b, c);
#else
			return x86_fnmsub_sse2(a, b, c);
#endif
		}
	}

#ifdef DPM_HAS_SSE2
	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fmadd(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::x86_fmadd(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fmsub(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::x86_fmsub(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fnmadd(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::x86_fnmadd(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fnmsub(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd<double, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			simd<double, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<double, detail::avec<N, A>>>; ++i)
			{
				ext::to_native_data(result)[i] = detail::x86_fnmsub(
						ext::to_native_data(a)[i],
						ext::to_native_data(b)[i],
						ext::to_native_data(c)[i]);
			}
			return result;
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<double, detail::avec<N, A>> fma(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b,
			const simd<double, detail::avec<N, A>> &c)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		return ext::fmadd(a, b, c);
	}
#endif
}

#endif