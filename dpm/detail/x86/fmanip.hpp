/*
 * Created by switchblade on 2023-02-05.
 */

#pragma once

#include "../fconst.hpp"
#include "mbase.hpp"

namespace dpm
{
	namespace detail
	{
		template<typename T, typename V>
		[[nodiscard]] DPM_FORCEINLINE V masksign(V x) noexcept { return bit_and(x, fill<V>(sign_bit<T>)); }
		template<typename T, typename V>
		[[nodiscard]] DPM_FORCEINLINE V copysign(V x, V m) noexcept { return bit_or(abs<T>(x), m); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC modf(__m128 x, __m128 *iptr) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC modf(__m128d x, __m128d *iptr) noexcept;

		[[nodiscard]] __m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128 x) noexcept;
		[[nodiscard]] __m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128d x) noexcept;

#ifdef DPM_HAS_AVX
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC modf(__m256 x, __m256 *iptr) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC modf(__m256d x, __m256d *iptr) noexcept;

		[[nodiscard]] __m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256 x) noexcept;
		[[nodiscard]] __m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256d x) noexcept;
#endif

#ifndef DPM_USE_SVML
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC logb(__m128 x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC logb(__m128d x) noexcept;

#ifdef DPM_HAS_AVX
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC logb(__m256 x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC logb(__m256d x) noexcept;
#endif
#else
		[[nodiscard]] DPM_FORCEINLINE __m128 logb(__m128 x) noexcept { return _mm_logb_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d logb(__m128d x) noexcept { return _mm_logb_pd(x); }

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 logb(__m256 x) noexcept { return _mm256_logb_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d logb(__m256d x) noexcept { return _mm256_logb_pd(x); }
#endif
#endif
#endif
	}

	/** Decomposes elements of vector \a x into integral and fractional parts, returning the fractional and storing the integral in \a iptr. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> modf(const detail::x86_simd<T, N, A> &x, detail::x86_simd<T, N, A> *iptr) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto &iptr) { res = detail::modf(x, iptr); }, result, x, *iptr);
		return result;
	}

	/** Extracts unbiased exponent of elements in vector \a x as integers, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<int, N, A> ilogb(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	{
		detail::x86_simd<int, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::ilogb(x); }, result, x);
		return result;
	}
	/** Extracts unbiased exponent of elements in vector \a x as floats, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> logb(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::logb(x); }, result, x);
		return result;
	}

	/** Copies sign bit from elements of vector \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &sign) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto s) { res = detail::copysign<T>(x, detail::masksign<T>(s)); }, result, x, sign);
		return result;
	}
	/** Copies sign bit from \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, T sign) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		const auto sign_vec = detail::masksign<T>(detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(sign));
		detail::vectorize([s = sign_vec](auto &res, auto x) { res = detail::copysign<T>(x, s); }, result, x);
		return result;
	}
}