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