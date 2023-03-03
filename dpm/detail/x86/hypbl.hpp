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
		[[nodiscard]] DPM_FORCEINLINE __m128 sinh(__m128 x) noexcept { return _mm_sinh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 cosh(__m128 x) noexcept { return _mm_cosh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 tanh(__m128 x) noexcept { return _mm_tanh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 asinh(__m128 x) noexcept { return _mm_asinh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 acosh(__m128 x) noexcept { return _mm_acosh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 atanh(__m128 x) noexcept { return _mm_atanh_ps(x); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d sinh(__m128d x) noexcept { return _mm_sinh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d cosh(__m128d x) noexcept { return _mm_cosh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d tanh(__m128d x) noexcept { return _mm_tanh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d asinh(__m128d x) noexcept { return _mm_asinh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d acosh(__m128d x) noexcept { return _mm_acosh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d atanh(__m128d x) noexcept { return _mm_atanh_pd(x); }
#endif
#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 sinh(__m256 x) noexcept { return _mm256_sinh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 cosh(__m256 x) noexcept { return _mm256_cosh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 tanh(__m256 x) noexcept { return _mm256_tanh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 asinh(__m256 x) noexcept { return _mm256_asinh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 acosh(__m256 x) noexcept { return _mm256_acosh_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 atanh(__m256 x) noexcept { return _mm256_atanh_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d sinh(__m256d x) noexcept { return _mm256_sinh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d cosh(__m256d x) noexcept { return _mm256_cosh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d tanh(__m256d x) noexcept { return _mm256_tanh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d asinh(__m256d x) noexcept { return _mm256_asinh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d acosh(__m256d x) noexcept { return _mm256_acosh_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d atanh(__m256d x) noexcept { return _mm256_atanh_pd(x); }
#endif
	}

	/** Calculates hyperbolic sine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> sinh(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::sinh(x); }, result, x);
		return result;
	}
	/** Calculates hyperbolic cosine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> cosh(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::cosh(x); }, result, x);
		return result;
	}
	/** Calculates hyperbolic tangent of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> tanh(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::tanh(x); }, result, x);
		return result;
	}
	/** Calculates hyperbolic arc-sine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> asinh(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::asinh(x); }, result, x);
		return result;
	}
	/** Calculates hyperbolic arc-cosine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> acosh(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::acosh(x); }, result, x);
		return result;
	}
	/** Calculates hyperbolic arc-tangent of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> atanh(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::atanh(x); }, result, x);
		return result;
	}
}

#endif