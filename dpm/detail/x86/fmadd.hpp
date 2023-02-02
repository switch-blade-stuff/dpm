/*
 * Created by switchblade on 2023-01-15.
 */

#pragma once

#include "type.hpp"

namespace dpm
{
	namespace detail
	{
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128 fmadd_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_add_ps(_mm_mul_ps(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128 fmsub_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_sub_ps(_mm_mul_ps(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128 fnmadd_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_sub_ps(c, _mm_mul_ps(a, b));
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m128 fnmsub_sse(__m128 a, __m128 b, __m128 c) noexcept
		{
			return _mm_sub_ps(_mm_setzero_ps(), fmadd_sse(a, b, c));
		}

		[[nodiscard]] DPM_FORCEINLINE __m128 fmadd(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmadd_ps(a, b, c);
#else
			return fmadd_sse(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 fmsub(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fmsub_ps(a, b, c);
#else
			return fmsub_sse(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 fnmadd(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmadd_ps(a, b, c);
#else
			return fnmadd_sse(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 fnmsub(__m128 a, __m128 b, __m128 c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm_fnmsub_ps(a, b, c);
#else
			return fnmsub_sse(a, b, c);
#endif
		}

#ifdef DPM_HAS_SSE2
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
#endif

#ifdef DPM_HAS_AVX
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256 fmadd_avx(__m256 a, __m256 b, __m256 c) noexcept
		{
			return _mm256_add_ps(_mm256_mul_ps(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256 fmsub_avx(__m256 a, __m256 b, __m256 c) noexcept
		{
			return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256 fnmadd_avx(__m256 a, __m256 b, __m256 c) noexcept
		{
			return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256 fnmsub_avx(__m256 a, __m256 b, __m256 c) noexcept
		{
			return _mm256_sub_ps(_mm256_setzero_ps(), fmadd_avx(a, b, c));
		}

		[[nodiscard]] DPM_FORCEINLINE __m256 fmadd(__m256 a, __m256 b, __m256 c) noexcept
		{
		#ifdef DPM_HAS_FMA
			return _mm256_fmadd_ps(a, b, c);
		#else
			return fmadd_avx(a, b, c);
		#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 fmsub(__m256 a, __m256 b, __m256 c) noexcept
		{
		#ifdef DPM_HAS_FMA
			return _mm256_fmsub_ps(a, b, c);
		#else
			return fmsub_avx(a, b, c);
		#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 fnmadd(__m256 a, __m256 b, __m256 c) noexcept
		{
		#ifdef DPM_HAS_FMA
			return _mm256_fnmadd_ps(a, b, c);
		#else
			return fnmadd_avx(a, b, c);
		#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 fnmsub(__m256 a, __m256 b, __m256 c) noexcept
		{
		#ifdef DPM_HAS_FMA
			return _mm256_fnmsub_ps(a, b, c);
		#else
			return fnmsub_avx(a, b, c);
		#endif
		}

		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d fmadd_avx(__m256d a, __m256d b, __m256d c) noexcept
		{
			return _mm256_add_pd(_mm256_mul_pd(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d fmsub_avx(__m256d a, __m256d b, __m256d c) noexcept
		{
			return _mm256_sub_pd(_mm256_mul_pd(a, b), c);
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d fnmadd_avx(__m256d a, __m256d b, __m256d c) noexcept
		{
			return _mm256_sub_pd(c, _mm256_mul_pd(a, b));
		}
		[[maybe_unused]] [[nodiscard]] DPM_FORCEINLINE __m256d fnmsub_avx(__m256d a, __m256d b, __m256d c) noexcept
		{
			return _mm256_sub_pd(_mm256_setzero_pd(), fmadd_avx(a, b, c));
		}

		[[nodiscard]] DPM_FORCEINLINE __m256d fmadd(__m256d a, __m256d b, __m256d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm256_fmadd_pd(a, b, c);
#else
			return fmadd_avx(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d fmsub(__m256d a, __m256d b, __m256d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm256_fmsub_pd(a, b, c);
#else
			return fmsub_avx(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d fnmadd(__m256d a, __m256d b, __m256d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm256_fnmadd_pd(a, b, c);
#else
			return fnmadd_avx(a, b, c);
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d fnmsub(__m256d a, __m256d b, __m256d c) noexcept
		{
#ifdef DPM_HAS_FMA
			return _mm256_fnmsub_pd(a, b, c);
#else
			return fnmsub_avx(a, b, c);
#endif
		}
#endif
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmadd(
				const detail::x86_simd<T, N, A> &a,
				const detail::x86_simd<T, N, A> &b,
				const detail::x86_simd<T, N, A> &c)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fmadd(a, b, c); }, result, a, b, c);
			return result;
		}
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmsub(
				const detail::x86_simd<T, N, A> &a,
				const detail::x86_simd<T, N, A> &b,
				const detail::x86_simd<T, N, A> &c)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fmsub(a, b, c); }, result, a, b, c);
			return result;
		}
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fnmadd(
				const detail::x86_simd<T, N, A> &a,
				const detail::x86_simd<T, N, A> &b,
				const detail::x86_simd<T, N, A> &c)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fnmadd(a, b, c); }, result, a, b, c);
			return result;
		}
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fnmsub(
				const detail::x86_simd<T, N, A> &a,
				const detail::x86_simd<T, N, A> &b,
				const detail::x86_simd<T, N, A> &c)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fnmsub(a, b, c); }, result, a, b, c);
			return result;
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fma(
			const detail::x86_simd<T, N, A> &a,
			const detail::x86_simd<T, N, A> &b,
			const detail::x86_simd<T, N, A> &c)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		return ext::fmadd(a, b, c);
	}
}