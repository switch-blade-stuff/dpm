/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "math_fwd.hpp"
#include "type.hpp"

#ifdef DPM_HAS_SSE2

#include "mbase.hpp"
#include "class.hpp"

namespace dpm
{
	namespace detail
	{
#ifndef DPM_USE_SVML
		[[nodiscard]] std::pair<__m128, __m128> DPM_PUBLIC DPM_MATHFUNC sincos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC sin(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC cos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC tan(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC cot(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC asin(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC acos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC atan(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC atan2(__m128 a, __m128 b) noexcept;

		[[nodiscard]] std::pair<__m128d, __m128d> DPM_PUBLIC DPM_MATHFUNC sincos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC sin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC cos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC tan(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC cot(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC asin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC acos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC atan(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC atan2(__m128d a, __m128d b) noexcept;

#ifdef DPM_HAS_AVX
		[[nodiscard]] std::pair<__m256, __m256> DPM_PUBLIC DPM_MATHFUNC sincos(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC sin(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC cos(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC tan(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC cot(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC asin(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC acos(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC atan(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC atan2(__m256 a, __m256 b) noexcept;

		[[nodiscard]] std::pair<__m256d, __m256d> DPM_PUBLIC DPM_MATHFUNC sincos(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC sin(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC cos(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC tan(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC cot(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC asin(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC acos(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC atan(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC atan2(__m256d a, __m256d b) noexcept;
#endif
#else
		[[nodiscard]] DPM_FORCEINLINE std::pair<__m128, __m128> sincos(__m128 x) noexcept
		{
			__m128 sin, cos;
			sin = _mm_sincos_ps(&cos, x);
			return {sin, cos};
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 sin(__m128 x) noexcept { return _mm_sin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 cos(__m128 x) noexcept { return _mm_cos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 tan(__m128 x) noexcept { return _mm_tan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 cot(__m128 x) noexcept { return _mm_rcp_ps(_mm_tan_ps(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m128 asin(__m128 x) noexcept { return _mm_asin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 acos(__m128 x) noexcept { return _mm_acos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 atan(__m128 x) noexcept { return _mm_atan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 atan2(__m128 a, __m128 b) noexcept { return _mm_atan2_ps(a, b); }

		[[nodiscard]] DPM_FORCEINLINE std::pair<__m128d, __m128d> sincos(__m128d x) noexcept
		{
			__m128d sin, cos;
			sin = _mm_sincos_pd(&cos, x);
			return {sin, cos};
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d sin(__m128d x) noexcept { return _mm_sin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d cos(__m128d x) noexcept { return _mm_cos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d tan(__m128d x) noexcept { return _mm_tan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d cot(__m128d x) noexcept
		{
			__m128d sin, cos;
			sin = _mm_sincos_pd(&cos, x);
			return _mm_div_pd(cos, sin);
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d asin(__m128d x) noexcept { return _mm_asin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d acos(__m128d x) noexcept { return _mm_acos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d atan(__m128d x) noexcept { return _mm_atan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d atan2(__m128d a, __m128d b) noexcept { return _mm_atan2_pd(a, b); }

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE std::pair<__m256, __m256> sincos(__m256 x) noexcept
		{
			__m256 sin, cos;
			sin = _mm_sincos_ps(&cos, x);
			return {sin, cos};
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 sin(__m256 x) noexcept { return _mm256_sin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 cos(__m256 x) noexcept { return _mm256_cos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 tan(__m256 x) noexcept { return _mm256_tan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 cot(__m256 x) noexcept { return _mm256_rcp_ps(_mm256_tan_ps(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m256 asin(__m256 x) noexcept { return _mm256_asin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 acos(__m256 x) noexcept { return _mm256_acos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 atan(__m256 x) noexcept { return _mm256_atan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 atan2(__m256 a, __m256 b) noexcept { return _mm256_atan2_ps(a, b); }

		[[nodiscard]] std::pair<__m256d, __m256d> DPM_FORCEINLINEsincos(__m256d x) noexcept
		{
			__m256d sin, cos;
			sin = _mm_sincos_ps(&cos, x);
			return {sin, cos};
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d sin(__m256d x) noexcept { return _mm256_sin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d cos(__m256d x) noexcept { return _mm256_cos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d tan(__m256d x) noexcept { return _mm256_tan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d cot(__m256d x) noexcept
		{
			__m256d sin, cos;
			sin = _mm256_sincos_pd(&cos, x);
			return _mm256_div_pd(cos, sin);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d asin(__m256d x) noexcept { return _mm256_asin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d acos(__m256d x) noexcept { return _mm256_acos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d atan(__m256d x) noexcept { return _mm256_atan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d atan2(__m256d a, __m256d b) noexcept { return _mm256_atan2_pd(a, b); }
#endif
#endif
	}

	/** Calculates sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> sin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::sin(x); }, result, x);
		return result;
	}
	/** Calculates cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> cos(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::cos(x); }, result, x);
		return result;
	}
	/** Calculates tangent of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> tan(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::tan(x); }, result, x);
		return result;
	}
	/** Calculates arc-sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> asin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::asin(x); }, result, x);
		return result;
	}
	/** Calculates arc-cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> acos(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::acos(x); }, result, x);
		return result;
	}
	/** Calculates arc-tangent of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> atan(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::atan(x); }, result, x);
		return result;
	}
	/** Calculates arc-tangent of quotient of elements in vectors \a a and \a b, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> atan2(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::atan2(a, b); }, result, a, b);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates sine and cosine of elements in vector \a x, and assigns results to elements of \a out_sin and \a out_cos respectively. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		DPM_FORCEINLINE void sincos(const detail::x86_simd<T, N, A> &x, detail::x86_simd<T, N, A> &out_sin, detail::x86_simd<T, N, A> &out_cos) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::vectorize([](auto x, auto &out_sin, auto &out_cos)
			                  {
				                  const auto [sin, cos] = detail::sincos(x);
				                  out_sin = sin;
				                  out_cos = cos;
			                  }, x, out_sin, out_cos);
		}

		/** Calculates cotangent of elements in vector \a x, and returns the resulting vector. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> cot(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::cot(x); }, result, x);
			return result;
		}
	}
}

#endif