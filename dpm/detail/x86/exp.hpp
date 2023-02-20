/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "mbase.hpp"

#ifndef DPM_USE_SVML

#include "transform.hpp"
#include "except.hpp"
#include "lut.hpp"

#endif

namespace dpm
{
	namespace detail
	{
#if defined(DPM_USE_SVML)
		[[nodiscard]] DPM_FORCEINLINE __m128 exp(__m128 x) noexcept { return _mm_exp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 exp2(__m128 x) noexcept { return _mm_exp2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 expm1(__m128 x) noexcept { return _mm_expm1_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m128 log(__m128 x) noexcept { return _mm_log_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 log2(__m128 x) noexcept { return _mm_log2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 log10(__m128 x) noexcept { return _mm_log10_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 log1p(__m128 x) noexcept { return _mm_log1p_ps(x); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d exp(__m128d x) noexcept { return _mm_exp_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d exp2(__m128d x) noexcept { return _mm_exp2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d expm1(__m128d x) noexcept { return _mm_expm1_pd(x); }

		[[nodiscard]] DPM_FORCEINLINE __m128d log(__m128d x) noexcept { return _mm_log_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d log2(__m128d x) noexcept { return _mm_log2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d log10(__m128d x) noexcept { return _mm_log10_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d log1p(__m128d x) noexcept { return _mm_log1p_pd(x); }
#endif
#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 exp(__m256 x) noexcept { return _mm256_exp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 exp2(__m256 x) noexcept { return _mm256_exp2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 expm1(__m256 x) noexcept { return _mm256_expm1_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256 log(__m256 x) noexcept { return _mm256_log_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 log2(__m256 x) noexcept { return _mm256_log2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 log10(__m256 x) noexcept { return _mm256_log10_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 log1p(__m256 x) noexcept { return _mm256_log1p_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d exp(__m256d x) noexcept { return _mm256_exp_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d exp2(__m256d x) noexcept { return _mm256_exp2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d expm1(__m256d x) noexcept { return _mm256_expm1_pd(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d log(__m256d x) noexcept { return _mm256_log_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d log2(__m256d x) noexcept { return _mm256_log2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d log10(__m256d x) noexcept { return _mm256_log10_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d log1p(__m256d x) noexcept { return _mm256_log1p_pd(x); }
#endif
#elif defined(DPM_HAS_SSE2)
		template<std::same_as<float> T, typename V, typename Vi>
		[[nodiscard]] DPM_FORCEINLINE std::pair<V, V> get_invc_logc(Vi i) noexcept
		{
			const auto data = std::span{logtab_f32};
			const auto i_invc = bit_shiftl<std::int32_t, 1>(i);
			const auto i_logc = add<std::int32_t>(i_invc, fill<Vi>(1));
			const auto v_invc = lut_load<V, std::int32_t>(data, i_invc);
			const auto v_logc = lut_load<V, std::int32_t>(data, i_logc);
			return {v_invc, v_logc};
		}
		template<std::same_as<double> T, typename V, typename Vi>
		[[nodiscard]] DPM_FORCEINLINE std::pair<V, V> get_invc_logc(Vi i) noexcept
		{
			const auto data = std::span{logtab_f64};
			const auto i_invc = bit_shiftl<std::int64_t, 1>(i);
			const auto i_logc = add<std::int64_t>(i_invc, fill<Vi>(1ull));
			const auto v_invc = lut_load<V, std::int64_t>(data, i_invc);
			const auto v_logc = lut_load<V, std::int64_t>(data, i_logc);
			return {v_invc, v_logc};
		}

		template<std::same_as<float> T, typename V, typename Vi = select_vector_t<std::int32_t, sizeof(V)>>
		[[nodiscard]] DPM_FORCEINLINE Vi log_normalize(V x) noexcept
		{
			constexpr std::int32_t sub_max = 0x7f00'0000;
			constexpr std::int32_t sub_off = 0x80'0000;
			const auto ix = std::bit_cast<Vi>(x);

			/* x is subnormal or not finite. */
			const auto is_subnorm = cmp_gt<std::int32_t>(sub<std::int32_t>(ix, fill<Vi>(sub_off)), fill<Vi>(sub_max - 1));
			/* Normalize x if it is subnormal. */
			const auto x_norm = mul<T>(x, fill<V>(0x1p23f));
			const auto ix_norm = sub<std::int32_t>(std::bit_cast<Vi>(x_norm), fill<Vi>(23 << 23));
			return blendv<std::int32_t>(ix, ix_norm, is_subnorm);
		}
		template<std::same_as<double> T, typename V, typename Vi = select_vector_t<std::int64_t, sizeof(V)>>
		[[nodiscard]] DPM_FORCEINLINE Vi log_normalize(V x) noexcept
		{
			constexpr std::int32_t sub_max = 0x7ff0 - 0x10;
			constexpr std::int32_t sub_off = 0x10;
			const auto ix = std::bit_cast<Vi>(x);

			/* x is subnormal, negative or not finite. */
			const auto top16 = bit_shiftr<std::int64_t, 48>(ix);
			const auto is_subnorm = cmp_gt_l32<std::int64_t>(sub<std::int64_t>(top16, fill<Vi>(sub_off)), fill<Vi>(sub_max - 1));
			/* Normalize x if it is subnormal. */
			const auto x_norm = mul<T>(x, fill<V>(0x1p52));
			const auto ix_norm = sub<std::int64_t>(std::bit_cast<Vi>(x_norm), fill<Vi>(52ull << 52));
			return blendv<std::int64_t>(ix, ix_norm, is_subnorm);
		}

		template<std::same_as<float> T, typename V, typename Vi = select_vector_t<std::int32_t, sizeof(V)>>
		[[nodiscard]] DPM_FORCEINLINE V log_excepts(V y, V x) noexcept
		{
#ifdef DPM_PROPAGATE_NAN
			/* log(inf) == inf; log(NaN) == NaN */
			const auto ix = std::bit_cast<Vi>(x);
			const auto not_finite = cmp_gt<std::int32_t>(ix, fill<Vi>(0x7f7f'ffff));
			y = blendv<T>(y, x, std::bit_cast<V>(not_finite));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* log(0) == inf + FE_DIVBYZERO */
			const auto zero_mask = cmp_eq<T>(x, setzero<V>());
			if (test_mask(zero_mask)) [[unlikely]] y = except_divzero<T>(y, zero_mask);
			/* log(-x) == NaN + FE_INVALID */
			const auto minus_mask = cmp_gt<std::int32_t>(setzero<Vi>(), ix);
			if (test_mask(minus_mask)) [[unlikely]] y = except_invalid<T>(y, std::bit_cast<V>(minus_mask));
#endif
			return y;
		}
		template<std::same_as<double> T, typename V, typename Vi = select_vector_t<std::int64_t, sizeof(V)>>
		[[nodiscard]] DPM_FORCEINLINE V log_excepts(V y, V x) noexcept
		{
#ifdef DPM_PROPAGATE_NAN
			/* log(inf) == inf; log(NaN) == NaN */
			const auto ix = std::bit_cast<Vi>(x);
			const auto not_finite = cmp_gt_h32<std::int64_t>(ix, fill<Vi>(0x7fef'ffff'0000'0000));
			y = blendv<T>(y, x, std::bit_cast<V>(not_finite));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* log(0) == inf + FE_DIVBYZERO */
			const auto zero_mask = cmp_eq<T>(x, setzero<V>());
			if (test_mask(zero_mask)) [[unlikely]] y = except_divzero<T>(y, zero_mask);
			/* log(-x) == NaN + FE_INVALID */
			const auto minus_mask = cmp_gt_h32<std::int64_t>(setzero<Vi>(), ix);
			if (test_mask(minus_mask)) [[unlikely]] y = except_invalid<T>(y, std::bit_cast<V>(minus_mask));
#endif
			return y;
		}

		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log2(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log10(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log1p(__m128 x) noexcept;


		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log2(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log10(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log1p(__m128d x) noexcept;

#ifdef DPM_HAS_AVX
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log2(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log10(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log1p(__m256 x) noexcept;

		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log2(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log10(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log1p(__m256d x) noexcept;
#endif
#endif
	}

#ifdef DPM_USE_SVML
	/** Raises *e* (Euler's number) to the power specified by elements of \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> exp(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::exp(x); }, result, x);
		return result;
	}
	/** Raises `2` to the power specified by elements of \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> exp2(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::exp(x); }, result, x);
		return result;
	}
	/** Raises *e* (Euler's number) to the power specified by elements of \a x, subtracts `1`, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> expm1(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::expm1(x); }, result, x);
		return result;
	}
#endif

#if defined(DPM_HAS_SSE2) || defined(DPM_USE_SVML)
	/** Calculates natural (base *e*) logarithm of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::log(x); }, result, x);
		return result;
	}
//	/** Calculates binary (base 2) logarithm of elements in vector \a x, and returns the resulting vector. */
//	template<std::floating_point T, std::size_t N, std::size_t A>
//	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log2(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && std::same_as<T, double>
//	{
//		detail::x86_simd<T, N, A> result = {};
//		detail::vectorize([](auto &res, auto x) { res = detail::log2(x); }, result, x);
//		return result;
//	}
//	/** Calculates common (base 10) logarithm of elements in vector \a x, and returns the resulting vector. */
//	template<std::floating_point T, std::size_t N, std::size_t A>
//	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log10(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && std::same_as<T, double>
//	{
//		detail::x86_simd<T, N, A> result = {};
//		detail::vectorize([](auto &res, auto x) { res = detail::log10(x); }, result, x);
//		return result;
//	}
//	/** Calculates natural (base *e*) logarithm of elements in vector \a x plus `1`, and returns the resulting vector. */
//	template<std::floating_point T, std::size_t N, std::size_t A>
//	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log1p(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && std::same_as<T, double>
//	{
//		detail::x86_simd<T, N, A> result = {};
//		detail::vectorize([](auto &res, auto x) { res = detail::log1p(x); }, result, x);
//		return result;
//	}
#endif
}
