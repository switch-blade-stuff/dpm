/*
 * Created by switchblade on 2023-01-11.
 */

#pragma once

#include "fmanip.hpp"

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 isinf(__m128 x) noexcept
		{
			const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
			return _mm_cmpeq_ps(abs(x), inf);
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 isfinite(__m128 x) noexcept
		{
			const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
#ifdef DPM_HAS_AVX
			return _mm_cmp_ps(abs(x), inf, _CMP_NEQ_OQ);
#else
			const auto abs_x = abs(x);
			return bit_and(_mm_cmpord_ps(abs_x, inf), _mm_cmpneq_ps(abs_x, inf));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 isnormal(__m128 x) noexcept
		{
			const auto abs_x = abs(x);
			const auto norm_min = _mm_set1_ps(std::bit_cast<float>(0x00800000));
			const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
			return bit_and(_mm_cmpnle_ps(inf, abs_x), _mm_cmpnlt_ps(abs_x, norm_min));
		}

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d isinf(__m128d x) noexcept
		{
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
			return _mm_cmpeq_pd(abs(x), inf);
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d isfinite(__m128d x) noexcept
		{
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
#ifdef DPM_HAS_AVX
			return _mm_cmp_pd(abs(x), inf, _CMP_NEQ_OQ);
#else
			const auto abs_x = abs(x);
			return bit_and(_mm_cmpord_pd(abs_x, inf), _mm_cmpneq_pd(abs_x, inf));
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d isnormal(__m128d x) noexcept
		{
			const auto abs_x = abs(x);
			const auto norm_min = _mm_set1_pd(std::bit_cast<double>(0x10'0000'0000'0000));
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
			return bit_and(_mm_cmpnle_pd(inf, abs_x), _mm_cmpnlt_pd(abs_x, norm_min));
		}
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 isinf(__m256 x) noexcept
		{
			const auto inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
			return cmp_eq<float>(abs(x), inf);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 isfinite(__m256 x) noexcept
		{
			const auto inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
			return _mm256_cmp_ps(abs(x), inf, _CMP_NEQ_OQ);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 isnormal(__m256 x) noexcept
		{
			const auto abs_x = abs(x);
			const auto norm_min = _mm256_set1_ps(std::bit_cast<float>(0x00800000));
			const auto inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
			return bit_and(_mm256_cmp_ps(inf, abs_x, _CMP_NLE_UQ), _mm256_cmp_ps(abs_x, norm_min, _CMP_NLT_UQ));
		}

		[[nodiscard]] DPM_FORCEINLINE __m256d isinf(__m256d x) noexcept
		{
			const auto inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
			return cmp_eq<double>(abs(x), inf);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d isfinite(__m256d x) noexcept
		{
			const auto inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
			return _mm256_cmp_pd(abs(x), inf, _CMP_NEQ_OQ);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d isnormal(__m256d x) noexcept
		{
			const auto abs_x = abs(x);
			const auto norm_min = _mm256_set1_pd(std::bit_cast<double>(0x10'0000'0000'0000));
			const auto inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
			return bit_and(_mm256_cmp_pd(inf, abs_x, _CMP_NLE_UQ), _mm256_cmp_pd(abs_x, norm_min, _CMP_NLT_UQ));
		}
#endif
	}

	/** Determines is elements of \a x are finite and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isfinite(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::isfinite(x); }, result, x);
		return result;
	}
	/** Determines is elements of \a x are infinite and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isinf(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::isinf(x); }, result, x);
		return result;
	}
	/** Determines is elements of \a x are NaN and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isnan(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::isunord(x, x); }, result, x);
		return result;
	}

	/** Determines is elements of \a x are normal and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isnormal(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::isnormal(x); }, result, x);
		return result;
	}

	/** Extracts a vector mask filled with sign bits of elements from vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> signbit(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::cmp_ne<T>(detail::masksign(x), detail::setzero<decltype(x)>()); }, result, x);
		return result;
	}

	/** Determines is elements of \a a are greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isgreater(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_lt<T>(b, a); }, result, a, b);
		return result;
	}
	/** Determines is elements of \a a are greater than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isgreaterequal(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_le<T>(b, a); }, result, a, b);
		return result;
	}
	/** Determines is elements of \a a are less than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isless(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_gt<T>(b, a); }, result, a, b);
		return result;
	}
	/** Determines is elements of \a a are less than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> islessequal(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_ge<T>(b, a); }, result, a, b);
		return result;
	}
	/** Determines is elements of \a a are less than or greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> islessgreater(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_or(detail::cmp_le<T>(b, a), detail::cmp_ge<T>(b, a)); }, result, a, b);
		return result;
	}
	/** Determines is either elements of \a a or \a b are unordered and returns the resulting mask. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> isunordered(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::isunord(a, b); }, result, a, b);
		return result;
	}
}