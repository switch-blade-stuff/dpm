/*
 * Created by switchblade on 2023-01-15.
 */

#pragma once

#include "type.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm
{
	namespace detail
	{
		[[maybe_unused]] [[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fmadd_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_add_ps(_mm_mul_ps(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fmsub_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_sub_ps(_mm_mul_ps(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fnmadd_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_sub_ps(c, _mm_mul_ps(a, b));
		}
		[[maybe_unused]] [[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fnmsub_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_sub_ps(_mm_setzero_ps(), x86_fmadd_sse(a, b, c));
		}

		[[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fmadd(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmadd_ps(a, b, c);
#else
			return x86_fmadd_sse(a, b, c);
#endif
		}
		[[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fmsub(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmsub_ps(a, b, c);
#else
			return x86_fmsub_sse(a, b, c);
#endif
		}
		[[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fnmadd(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmadd_ps(a, b, c);
#else
			return x86_fnmadd_sse(a, b, c);
#endif
		}
		[[nodiscard]] inline __m128 DPM_FORCEINLINE x86_fnmsub(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmsub_ps(a, b, c);
#else
			return x86_fnmsub_sse(a, b, c);
#endif
		}
	}

#ifdef DPM_HAS_SSE
	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<float, detail::avec<N, A>> fmadd(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd<float, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<float, N, A>
		{
			simd<float, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
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
		[[nodiscard]] inline simd<float, detail::avec<N, A>> fmsub(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd<float, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<float, N, A>
		{
			simd<float, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
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
		[[nodiscard]] inline simd<float, detail::avec<N, A>> fnmadd(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd<float, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<float, N, A>
		{
			simd<float, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
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
		[[nodiscard]] inline simd<float, detail::avec<N, A>> fnmsub(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd<float, detail::avec<N, A>> &c)
		noexcept requires detail::x86_overload_m128<float, N, A>
		{
			simd<float, detail::avec<N, A>> result;
			for (std::size_t i = 0; i < ext::native_data_size_v<simd<float, detail::avec<N, A>>>; ++i)
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
	[[nodiscard]] inline simd<float, detail::avec<N, A>> fma(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b,
			const simd<float, detail::avec<N, A>> &c)
	noexcept requires detail::x86_overload_m128<float, N, A>
	{
		return ext::fmadd(a, b, c);
	}
#endif
}

#endif