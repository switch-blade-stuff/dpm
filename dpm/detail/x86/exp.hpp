/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "mbase.hpp"

#ifndef DPM_USE_SVML

#include "transform.hpp"
#include "except.hpp"
#include "class.hpp"
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
		/* Use source_location to properly report assertions. */
		template<typename V, typename Vi, typename T, std::size_t N, typename I = int_of_size_t<sizeof(T)>>
		[[nodiscard]] DPM_FORCEINLINE std::pair<V, V> log_get_table(Vi idx, std::span<const T, N> table, std::source_location loc) noexcept
		{
			const auto i_invc = bit_shiftl<I, 1>(idx);
			const auto i_logc = add<I>(i_invc, fill<Vi>(static_cast<I>(1)));
			const auto v_invc = lut_load<V, I>(table, i_invc, loc);
			const auto v_logc = lut_load<V, I>(table, i_logc, loc);
			return {v_invc, v_logc};
		}
		template<typename T, typename V>
		[[nodiscard]] DPM_FORCEINLINE V log_excepts(V y, V x) noexcept
		{
#ifdef DPM_PROPAGATE_NAN
			/* log(inf) == inf; log(NaN) == NaN */
			y = blendv<T>(x, y, isfinite_abs(x));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* log(0) == inf + FE_DIVBYZERO */
			const auto zero_mask = cmp_eq<T>(x, setzero<V>());
			if (test_mask(zero_mask)) [[unlikely]] y = except_divzero<T>(y, x, zero_mask);
			/* log(-x) == NaN + FE_INVALID */
			const auto minus_mask = cmp_lt<T>(x, setzero<V>());
			if (test_mask(minus_mask)) [[unlikely]] y = except_invalid<T>(y, x, std::bit_cast<V>(minus_mask));
#endif
			return y;
		}

		template<std::same_as<float> T, typename V, typename Vi = select_vector_t<std::int32_t, sizeof(V)>>
		[[nodiscard]] DPM_FORCEINLINE Vi log_normalize(V x) noexcept
		{
			constexpr std::int32_t sub_max = 0x7f00'0000;
			constexpr std::int32_t sub_off = 0x80'0000;
			auto ix = std::bit_cast<Vi>(x);

			/* x is subnormal or not finite. */
			const auto is_subnorm = cmp_gt<std::int32_t>(sub<std::int32_t>(ix, fill<Vi>(sub_off)), fill<Vi>(sub_max - 1));
			if (test_mask(is_subnorm)) [[unlikely]] /* Avoid overflow in x * 2^23 */
			{
				/* Normalize x. */
				const auto x_norm = mul<T>(x, fill<V>(0x1p23f));
				const auto ix_norm = sub<std::int32_t>(std::bit_cast<Vi>(x_norm), fill<Vi>(23 << 23));
				ix = blendv<std::int32_t>(ix, ix_norm, is_subnorm);
			}
			return ix;
		}
		template<std::same_as<double> T, typename V, typename Vi = select_vector_t<std::int64_t, sizeof(V)>>
		[[nodiscard]] DPM_FORCEINLINE Vi log_normalize(V x) noexcept
		{
			constexpr std::int32_t sub_max = 0x7fe0;
			constexpr std::int32_t sub_off = 0x10;
			auto ix = std::bit_cast<Vi>(x);

			/* x is subnormal, negative or not finite. */
			const auto top16 = bit_shiftr<std::int64_t, 48>(ix);
			const auto is_subnorm = cmp_gt_l32<std::int64_t>(sub<std::int64_t>(top16, fill<Vi>(sub_off)), fill<Vi>(sub_max - 1));
			if (test_mask(is_subnorm)) [[unlikely]] /* Avoid overflow in x * 2^52 */
			{
				/* Normalize x. */
				const auto x_norm = mul<T>(x, fill<V>(0x1p52));
				const auto ix_norm = sub<std::int64_t>(std::bit_cast<Vi>(x_norm), fill<Vi>(52ull << 52));
				ix = blendv<std::int64_t>(ix, ix_norm, is_subnorm);
			}
			return ix;
		}

		/* Vectorized versions of log(float) & log(double) based on implementation from the ARM optimized routines library
		 * https://github.com/ARM-software/optimized-routines license: MIT */
		template<std::same_as<float> T, typename Vi, typename I = std::int32_t, typename V = select_vector_t<T, sizeof(Vi)>>
		[[nodiscard]] DPM_FORCEINLINE V eval_log(Vi ix) noexcept
		{
			/* Load invc & logc constants from the lookup table. */
			const auto tmp = sub<I>(ix, fill<Vi>(0x3f33'0000));
			auto i = bit_shiftr<I, 23 - logtab_bits_f32>(tmp);
			i = bit_and(i, fill<Vi>((1 << logtab_bits_f32) - 1));
			const auto [invc, logc] = log_get_table<V>(i, logtab_v<T>, std::source_location::current());

			/* x = 2^k z; where z is in range [0x3f33'0000, 2 * 0x3f33'0000] and exact.  */
			const auto k = bit_ashiftr<I, 23>(tmp);
			const auto z = std::bit_cast<V>(sub<I>(ix, bit_and(tmp, fill<Vi>(0xff80'0000))));

			/* log(x) = log1p(z/c-1) + log(c) + k*ln2 */
			const auto r = fmsub(z, invc, fill<V>(1.0f));
			const auto y0 = fmadd(cvt<T, I>(k), fill<V>(ln2<T>), logc);

			/* Approximate log1p(r).  */
			const auto r2 = mul<T>(r, r);
			auto y = fmadd(fill<V>(logcoff_f32[1]), r, fill<V>(logcoff_f32[2]));
			y = fmadd(fill<V>(logcoff_f32[0]), r2, y);
			return fmadd(y, r2, add<T>(y0, r));
		}
		template<std::same_as<double> T, typename Vi, typename I = std::int64_t, typename V = select_vector_t<T, sizeof(Vi)>>
		[[nodiscard]] DPM_FORCEINLINE V eval_log(Vi ix) noexcept
		{
			/* Load invc & logc constants from the lookup table. */
			const auto tmp = sub<I>(ix, fill<Vi>(0x3fe6'9009'0000'0000));
			auto i = bit_shiftr<I, 52 - logtab_bits_f64>(tmp);
			i = bit_and(i, fill<Vi>((1 << logtab_bits_f64) - 1));
			const auto [invc, logc] = log_get_table<V>(i, logtab_v<T>, std::source_location::current());

			/* x = 2^k z; where z is in range [0x3fe6'9009'0000'0000, 2 * 0x3fe6'9009'0000'0000] and exact.  */
			const auto k = bit_ashiftr<I, 52>(tmp);
			const auto z = std::bit_cast<V>(sub<I>(ix, bit_and(tmp, fill<Vi>(0xfffull << 52))));

			/* log(x) = log1p(z/c-1) + log(c) + k*ln2.  */
			const auto r = fmadd(z, invc, fill<V>(-1.0));
			/* We only care about the bottom bits anyway. */
			const auto kf = cvt_i32_f64<V>(cvt_i64_i32(k));
			const auto y0 = fmadd(kf, fill<V>(ln2<T>), logc);

			/* y = r2 * (logcoff_f64[0] + r * logcoff_f64[1] + r2 * (logcoff_f64[2] + r * logcoff_f64[3] + r2 * logcoff_f64[4])) + y0 + r  */
			const auto r2 = mul<T>(r, r);
			const auto p = fmadd(fill<V>(logcoff_f64[1]), r, fill<V>(logcoff_f64[0]));
			auto y = fmadd(fill<V>(logcoff_f64[3]), r, fill<V>(logcoff_f64[2]));
			y = fmadd(fill<V>(logcoff_f64[4]), r2, y);
			return fmadd(fmadd(y, r2, p), r2, add<T>(y0, r));
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
	/** Raises *e* (Euler's number) to the power specified by elements of \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> exp(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::exp(x); }, result, x);
		return result;
	}
	/** Raises `2` to the power specified by elements of \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> exp2(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::exp(x); }, result, x);
		return result;
	}
	/** Raises *e* (Euler's number) to the power specified by elements of \a x, and subtracts `1`. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> expm1(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::expm1(x); }, result, x);
		return result;
	}
#endif

#if defined(DPM_HAS_SSE2) || defined(DPM_USE_SVML)
	/** Calculates natural (base *e*) logarithm of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::log(x); }, result, x);
		return result;
	}
	/** Calculates binary (base 2) logarithm of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log2(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::log2(x); }, result, x);
		return result;
	}
	/** Calculates common (base 10) logarithm of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log10(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::log10(x); }, result, x);
		return result;
	}
	/** Calculates natural (base *e*) logarithm of elements in vector \a x plus `1`. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> log1p(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::log1p(x); }, result, x);
		return result;
	}
#endif
}
