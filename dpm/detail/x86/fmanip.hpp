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

#ifdef DPM_USE_SVML
		[[nodiscard]] DPM_FORCEINLINE __m128 logb(__m128 x) noexcept { return _mm_logb_ps(x); }
#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d logb(__m128d x) noexcept { return _mm_logb_pd(x); }
#endif
#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 logb(__m256 x) noexcept { return _mm256_logb_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d logb(__m256d x) noexcept { return _mm256_logb_pd(x); }
#endif
#endif

#ifdef DPM_HAS_SSE2
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC ldexp(__m128 x, __m128i exp) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC ldexp(__m128d x, __m128i exp) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m128 ldexp(__m128 x, const __m128i *exp, std::size_t i) noexcept { return ldexp(x, exp[i]); }
		[[nodiscard]] DPM_FORCEINLINE __m128d ldexp(__m128d x, const __m128i *exp, std::size_t i) noexcept
		{
			auto exp64 = exp[i / 2];
			if (const auto sign = _mm_srai_epi32(exp64, 31); i % 2)
				exp64 = _mm_unpackhi_epi32(exp64, sign);
			else
				exp64 = _mm_unpacklo_epi32(exp64, sign);
			return ldexp(x, exp64);
		}

		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC modf(__m128 x, __m128 *iptr) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC modf(__m128d x, __m128d *iptr) noexcept;

		[[nodiscard]] __m128i DPM_PUBLIC DPM_MATHFUNC ilogb(__m128 x) noexcept;
		[[nodiscard]] __m128i DPM_PUBLIC DPM_MATHFUNC ilogb(__m128d x) noexcept;

		DPM_FORCEINLINE void ilogb(__m128 x, __m128i *out, std::size_t i) noexcept { out[i] = ilogb(x); }
		DPM_FORCEINLINE void ilogb(__m128d x, __m128i *out, std::size_t i) noexcept
		{
			auto *out64 = reinterpret_cast<alias_t<std::uint64_t> *>(out);
			out64[i] = _mm_cvtsi128_si64(cvt<std::int32_t, std::int64_t>(ilogb(x)));
		}
		DPM_FORCEINLINE void ilogb2(__m128d x0, __m128d x1, __m128i *out, std::size_t i) noexcept
		{
			const auto i0 = ilogb(x0);
			const auto i1 = ilogb(x1);
			out[i] = pack_i64x2_i32(i0, i1);
		}

#ifndef DPM_USE_SVML
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC logb(__m128 x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC logb(__m128d x) noexcept;
#endif

		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC nextafter(__m128 from, __m128 to) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC nextafter(__m128d from, __m128d to) noexcept;
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC ldexp(__m256 x, __m256i exp) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC ldexp(__m256d x, __m256i exp) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m256 ldexp(__m256 x, const __m256i *exp, std::size_t i) noexcept { return ldexp(x, exp[i]); }
		[[nodiscard]] DPM_FORCEINLINE __m256d ldexp(__m256d x, const __m128i *exp, std::size_t i) noexcept
		{
			const auto xh = _mm256_castpd256_pd128(_mm256_unpackhi_pd(x, x));
			const auto xl = _mm256_castpd256_pd128(x);
			const auto h = ldexp(xh, exp, i * 2 + 1);
			const auto l = ldexp(xl, exp, i * 2);
			return _mm256_set_m128d(h, l);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d ldexp(__m256d x, const __m256i *exp, std::size_t i) noexcept
		{
#ifdef DPM_HAS_AVX2
			auto exp64 = exp[i / 2];
			if (const auto sign = _mm256_srai_epi32(exp64, 31); i % 2)
				exp64 = _mm256_unpackhi_epi32(exp64, sign);
			else
				exp64 = _mm256_unpacklo_epi32(exp64, sign);
			return ldexp(x, exp64);
#else
			return ldexp(x, reinterpret_cast<const __m128i *>(exp), i);
#endif
		}

		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC modf(__m256 x, __m256 *iptr) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC modf(__m256d x, __m256d *iptr) noexcept;

		[[nodiscard]] __m256i DPM_PUBLIC DPM_MATHFUNC ilogb(__m256 x) noexcept;
		[[nodiscard]] __m256i DPM_PUBLIC DPM_MATHFUNC ilogb(__m256d x) noexcept;

		DPM_FORCEINLINE void ilogb(__m256 x, __m256i *out, std::size_t i) noexcept { out[i] = ilogb(x); }
		DPM_FORCEINLINE void ilogb(__m256d x, __m256i *out, std::size_t i) noexcept
		{
			auto *out128 = reinterpret_cast<__m128i *>(out);
			out128[i] = cvt<std::int32_t, std::int64_t>(ilogb(x));
		}
		DPM_FORCEINLINE void ilogb2(__m256d x0, __m256d x1, __m256i *out, std::size_t i) noexcept
		{
			const auto i0 = ilogb(x0);
			const auto i1 = ilogb(x1);
			out[i] = pack_i64x2_i32(i0, i1);
		}

#ifndef DPM_USE_SVML
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC logb(__m256 x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC logb(__m256d x) noexcept;
#endif

		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC nextafter(__m256 from, __m256 to) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC nextafter(__m256d from, __m256d to) noexcept;
#endif
	}

#ifdef DPM_HAS_SSE2
	/** Multiplies elements of vector \a x by `2` raised to power specified by elements of vector \a exp, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> ldexp(const detail::x86_simd<T, N, A> &x, detail::x86_simd<int, N, A> exp) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto exp_data = ext::to_native_data(exp);
		const auto x_data = ext::to_native_data(x);
		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::ldexp(x_data[i], exp_data.data(), i);
		return result;
	}
	/** @copydoc ldexp */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> scalbn(const detail::x86_simd<T, N, A> &x, detail::x86_simd<int, N, A> exp) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	{
		return ldexp(x, exp);
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
		detail::x86_simd<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto x_data = ext::to_native_data(x);

		std::size_t i = 0;
		constexpr auto native_size = ext::native_data_size_v<detail::x86_simd<T, N, A>>;
		if constexpr (sizeof(T) > sizeof(int)) for (; i + 1 < native_size; i += 2)
				detail::ilogb2(x_data[i], x_data[i + 1], result_data.data(), i);
		for (; i < native_size; ++i) detail::ilogb(x_data[i], result_data.data(), i);

		return result;
	}

	/** Finds next representable value from elements of vector \a from to elements of vector \a to, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> nextafter(const detail::x86_simd<T, N, A> &from, const detail::x86_simd<T, N, A> &to) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto from, auto to) { res = detail::nextafter(from, to); }, result, from, to);
		return result;
	}
#endif

#if defined(DPM_USE_SVML) || defined(DPM_HAS_SSE2)
	/** Extracts unbiased exponent of elements in vector \a x as floats, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> logb(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::logb(x); }, result, x);
		return result;
	}
#endif

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