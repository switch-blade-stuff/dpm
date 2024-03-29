/*
 * Created by switchblade on 2023-01-11.
 */

#pragma once

#include "fmanip.hpp"

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 isinf_abs(__m128 x) noexcept
		{
			return _mm_cmpeq_ps(x, _mm_set1_ps(std::numeric_limits<float>::infinity()));
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 isfinite_abs(__m128 x) noexcept
		{
			const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
		#ifdef DPM_HAS_AVX
			return _mm_cmp_ps(x, inf, _CMP_NEQ_OQ);
		#else
			return _mm_cmplt_ps(x, inf); /* NaN < inf == false */
		#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128 isnormal_abs(__m128 x) noexcept
		{
			const auto norm_min = _mm_set1_ps(std::bit_cast<float>(0x00800000));
			const auto inf = _mm_set1_ps(std::numeric_limits<float>::infinity());
			return bit_and(_mm_cmpnle_ps(inf, x), _mm_cmpnlt_ps(x, norm_min));
		}

		[[nodiscard]] DPM_FORCEINLINE __m128 isinf(__m128 x) noexcept { return isinf_abs(abs<float>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m128 isfinite(__m128 x) noexcept { return isfinite_abs(abs<float>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m128 isnormal(__m128 x) noexcept { return isnormal_abs(abs<float>(x)); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d isinf_abs(__m128d x) noexcept
		{
			return _mm_cmpeq_pd(x, _mm_set1_pd(std::numeric_limits<double>::infinity()));
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d isfinite_abs(__m128d x) noexcept
		{
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
#ifdef DPM_HAS_AVX
			return _mm_cmp_pd(x, inf, _CMP_NEQ_OQ);
#else
			return _mm_cmplt_pd(x, inf); /* NaN < inf == false */
#endif
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d isnormal_abs(__m128d x) noexcept
		{
			const auto norm_min = _mm_set1_pd(std::bit_cast<double>(0x10'0000'0000'0000));
			const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
			return bit_and(_mm_cmpnle_pd(inf, x), _mm_cmpnlt_pd(x, norm_min));
		}

		[[nodiscard]] DPM_FORCEINLINE __m128d isinf(__m128d x) noexcept { return isinf_abs(abs<double>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m128d isfinite(__m128d x) noexcept { return isfinite_abs(abs<double>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m128d isnormal(__m128d x) noexcept { return isnormal_abs(abs<double>(x)); }
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 isinf_abs(__m256 x) noexcept
		{
			return cmp_eq<float>(x, _mm256_set1_ps(std::numeric_limits<float>::infinity()));
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 isfinite_abs(__m256 x) noexcept
		{
			return _mm256_cmp_ps(x, _mm256_set1_ps(std::numeric_limits<float>::infinity()), _CMP_NEQ_OQ);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256 isnormal_abs(__m256 x) noexcept
		{
			const auto norm_min = _mm256_set1_ps(std::bit_cast<float>(0x00800000));
			const auto inf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
			return bit_and(_mm256_cmp_ps(inf, x, _CMP_NLE_UQ), _mm256_cmp_ps(x, norm_min, _CMP_NLT_UQ));
		}

		[[nodiscard]] DPM_FORCEINLINE __m256 isinf(__m256 x) noexcept { return isinf_abs(abs<float>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m256 isfinite(__m256 x) noexcept { return isfinite_abs(abs<float>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m256 isnormal(__m256 x) noexcept { return isnormal_abs(abs<float>(x)); }

		[[nodiscard]] DPM_FORCEINLINE __m256d isinf_abs(__m256d x) noexcept
		{
			return cmp_eq<double>(x, _mm256_set1_pd(std::numeric_limits<double>::infinity()));
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d isfinite_abs(__m256d x) noexcept
		{
			return _mm256_cmp_pd(x, _mm256_set1_pd(std::numeric_limits<double>::infinity()), _CMP_NEQ_OQ);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d isnormal_abs(__m256d x) noexcept
		{
			const auto norm_min = _mm256_set1_pd(std::bit_cast<double>(0x10'0000'0000'0000));
			const auto inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());
			return bit_and(_mm256_cmp_pd(inf, x, _CMP_NLE_UQ), _mm256_cmp_pd(x, norm_min, _CMP_NLT_UQ));
		}

		[[nodiscard]] DPM_FORCEINLINE __m256d isinf(__m256d x) noexcept { return isinf_abs(abs<double>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m256d isfinite(__m256d x) noexcept { return isfinite_abs(abs<double>(x)); }
		[[nodiscard]] DPM_FORCEINLINE __m256d isnormal(__m256d x) noexcept { return isnormal_abs(abs<double>(x)); }
#endif
	}

	/** Determines if elements of \a x are NaN. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isnan(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(isnan(packed));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::isunord(x, x); }, result, x);
			return result;
		}
	}
	/** Determines if elements of \a x are infinite. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isinf(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(isinf(packed));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::isinf(x); }, result, x);
			return result;
		}
	}
	/** Determines if elements of \a x are finite. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isfinite(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(isfinite(packed));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::isfinite(x); }, result, x);
			return result;
		}
	}
	/** Determines if elements of \a x are normal. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isnormal(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(isnormal(packed));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::isnormal(x); }, result, x);
			return result;
		}
	}
	/** Extracts a vector mask filled with sign bits of elements from vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> signbit(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_mask<T, N, A>>(signbit(packed));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::masksign<T>(x); }, result, x);
			return result;
		}
	}

	/** Determines if elements of \a a are greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isgreater(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(isgreater(packed_a, packed_b));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_lt<T>(b, a); }, result, a, b);
			return result;
		}
	}
	/** Determines if elements of \a a are greater than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isgreaterequal(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(isgreaterequal(packed_a, packed_b));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_le<T>(b, a); }, result, a, b);
			return result;
		}
	}
	/** Determines if elements of \a a are less than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isless(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(isless(packed_a, packed_b));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_gt<T>(b, a); }, result, a, b);
			return result;
		}
	}
	/** Determines if elements of \a a are less than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> islessequal(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(islessequal(packed_a, packed_b));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::cmp_ge<T>(b, a); }, result, a, b);
			return result;
		}
	}
	/** Determines if elements of \a a are less than or greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> islessgreater(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(islessgreater(packed_a, packed_b));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::bit_or(detail::cmp_le<T>(b, a), detail::cmp_ge<T>(b, a)); }, result, a, b);
			return result;
		}
	}
	/** Determines is either elements of \a a or \a b are unordered. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_mask<T, N, A> isunordered(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_mask<T, N, A>>(isunordered(packed_a, packed_b));
		}
		else
		{
			detail::x86_mask <T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::isunord(a, b); }, result, a, b);
			return result;
		}
	}
}