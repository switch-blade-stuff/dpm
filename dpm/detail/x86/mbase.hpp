/*
 * Created by switchblade on 2023-02-01.
 */

#pragma once

#include "math_fwd.hpp"
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
			const auto zero = setzero<__m128>();
			return _mm_sub_ps(zero, fmadd_sse(a, b, c));
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

		[[nodiscard]] DPM_FORCEINLINE __m128 isunord(__m128 a, __m128 b) noexcept { return _mm_cmpunord_ps(a, b); }

		template<std::same_as<float> T>
		[[nodiscard]] DPM_FORCEINLINE __m128 abs(__m128 x) noexcept { return _mm_and_ps(x, _mm_set1_ps(std::bit_cast<T>(0x7fff'ffff))); }

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
			const auto tmp = _mm_undefined_pd();
			const auto zero = _mm_xor_pd(tmp, tmp);
			return _mm_sub_pd(zero, fmadd_sse(a, b, c));
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

		[[nodiscard]] DPM_FORCEINLINE __m128d isunord(__m128d a, __m128d b) noexcept { return _mm_cmpunord_pd(a, b); }

		template<std::same_as<double> T>
		[[nodiscard]] DPM_FORCEINLINE __m128d abs(__m128d x) noexcept { return _mm_and_pd(x, _mm_set1_pd(std::bit_cast<T>(0x7fff'ffff'ffff'ffff))); }
		template<std::integral T, typename V>
		[[nodiscard]] DPM_FORCEINLINE V abs(V x) noexcept
		{
			const auto t = bit_shiftr<T, std::numeric_limits<T>::digits - 1>(x);
			return sub<T>(bit_xor(x, t), t);
		}

#ifdef DPM_HAS_SSSE3
		template<integral_of_size<1> T>
		[[nodiscard]] DPM_FORCEINLINE __m128i abs(__m128i x) noexcept { return _mm_abs_epi8(x); }
		template<integral_of_size<2> T>
		[[nodiscard]] DPM_FORCEINLINE __m128i abs(__m128i x) noexcept { return _mm_abs_epi16(x); }
		template<integral_of_size<4> T>
		[[nodiscard]] DPM_FORCEINLINE __m128i abs(__m128i x) noexcept { return _mm_abs_epi32(x); }
#endif
#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
		template<integral_of_size<8> T>
		[[nodiscard]] DPM_FORCEINLINE __m128i abs(__m128i x) noexcept { return _mm_abs_epi64(x); }
#endif
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
			const auto tmp = _mm256_undefined_ps();
			const auto zero = _mm256_xor_ps(tmp, tmp);
			return _mm256_sub_ps(zero, fmadd_avx(a, b, c));
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
			const auto tmp = _mm256_undefined_pd();
			const auto zero = _mm256_xor_pd(tmp, tmp);
			return _mm256_sub_pd(zero, fmadd_avx(a, b, c));
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

		[[nodiscard]] DPM_FORCEINLINE __m256 isunord(__m256 a, __m256 b) noexcept { return _mm256_cmp_ps(a, b, _CMP_UNORD_Q); }
		[[nodiscard]] DPM_FORCEINLINE __m256d isunord(__m256d a, __m256d b) noexcept { return _mm256_cmp_pd(a, b, _CMP_UNORD_Q); }

		template<std::same_as<float> T>
		[[nodiscard]] DPM_FORCEINLINE __m256 abs(__m256 x) noexcept { return _mm256_and_ps(x, _mm256_set1_ps(std::bit_cast<T>(0x7fff'ffff))); }
		template<std::same_as<double> T>
		[[nodiscard]] DPM_FORCEINLINE __m256d abs(__m256d x) noexcept { return _mm256_and_pd(x, _mm256_set1_pd(std::bit_cast<T>(0x7fff'ffff'ffff'ffff))); }

#ifdef DPM_HAS_AVX2
		template<integral_of_size<1> T>
		[[nodiscard]] DPM_FORCEINLINE __m256i abs(__m256i x) noexcept { return _mm256_abs_epi8(x); }
		template<integral_of_size<2> T>
		[[nodiscard]] DPM_FORCEINLINE __m256i abs(__m256i x) noexcept { return _mm256_abs_epi16(x); }
		template<integral_of_size<4> T>
		[[nodiscard]] DPM_FORCEINLINE __m256i abs(__m256i x) noexcept { return _mm256_abs_epi32(x); }

#if defined(DPM_HAS_AVX512F) && defined(DPM_HAS_AVX512LV)
		template<integral_of_size<8> T>
		[[nodiscard]] DPM_FORCEINLINE __m256i abs(__m256i x) noexcept { return _mm256_abs_epi64(x); }
#endif
#endif
#endif
	}

	/** Calculates absolute value of elements in vector \a x. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> abs(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x) { res = detail::abs<T>(x); }, result, x);
		return result;
	}
	/** @copydoc abs */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fabs(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> { return abs(x); }

	/** Calculates the maximum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmax(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::blendv<T>(detail::max<T>(a, b), b, detail::isunord(a)); }, result, a, b);
		return result;
	}
	/** Calculates the minimum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmin(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b) { res = detail::blendv<T>(detail::min<T>(a, b), b, detail::isunord(a)); }, result, a, b);
		return result;
	}
	/** Returns the positive difference between elements of vectors \a a and \a b. Equivalent to `max(simd<T, Abi>{0}, a - b)`. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fdim(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([]<typename V>(V &res, V a, V b) { res = detail::max<T>(detail::setzero<V>(), detail::sub<T>(a, b)); }, result, a, b);
		return result;
	}

	/** Calculates the maximum of elements in \a a and scalar \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmax(const detail::x86_simd<T, N, A> &a, T b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
		detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::blendv<T>(detail::max<T>(a, b), b, detail::isunord(a)); }, result, a);
		return result;
	}
	/** Calculates the minimum of elements in \a a and scalar \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmin(const detail::x86_simd<T, N, A> &a, T b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
		detail::vectorize([b = b_vec](auto &res, auto a) { res = detail::blendv<T>(detail::min<T>(a, b), b, detail::isunord(a)); }, result, a);
		return result;
	}
	/** Returns the positive difference between elements of vectors \a a and scalar \a b. Equivalent to `max(simd<T, Abi>{0}, a - b)`. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fdim(const detail::x86_simd<T, N, A> &a, T b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		const auto b_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(b);
		detail::vectorize([b = b_vec]<typename V>(V &res, V a) { res = detail::max<T>(detail::setzero<V>(), detail::sub<T>(a, b)); }, result, a);
		return result;
	}

	/** Preforms linear interpolation or extrapolation between elements of vectors \a a and \a b using factor \a f */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> lerp(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, const detail::x86_simd<T, N, A> &f) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto a, auto b, auto f) { res = detail::fmadd(detail::sub<T>(b, a), f, a); }, result, a, b, f);
		return result;
	}
	/** Preforms linear interpolation or extrapolation between elements of vectors \a a and \a b using factor \a f */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> lerp(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, T f) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		const auto f_vec = detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(f);
		detail::vectorize([f = f_vec](auto &res, auto a, auto b) { res = detail::fmadd(detail::sub<T>(b, a), f, a); }, result, a, b);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmadd(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, const detail::x86_simd<T, N, A> &c) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fmadd(a, b, c); }, result, a, b, c);
			return result;
		}
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fmsub(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, const detail::x86_simd<T, N, A> &c) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fmsub(a, b, c); }, result, a, b, c);
			return result;
		}
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fnmadd(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, const detail::x86_simd<T, N, A> &c) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fnmadd(a, b, c); }, result, a, b, c);
			return result;
		}
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fnmsub(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, const detail::x86_simd<T, N, A> &c) noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b, auto c) { res = detail::fnmsub(a, b, c); }, result, a, b, c);
			return result;
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> fma(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b, const detail::x86_simd<T, N, A> &c) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return ext::fmadd(a, b, c);
	}
}