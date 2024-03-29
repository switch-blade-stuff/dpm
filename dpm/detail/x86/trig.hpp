/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "math_fwd.hpp"
#include "type.hpp"

#if defined(DPM_ARCH_X86)

#include "except.hpp"
#include "polevl.hpp"
#include "class.hpp"

namespace dpm
{
	namespace detail
	{
		template<typename T>
		struct sincos_ret { T sin, cos; };

#if !defined(DPM_USE_SVML) && defined(DPM_HAS_SSE2)
		enum sincos_op
		{
			OP_SINCOS = 3,
			OP_COS = 2,
			OP_SIN = 1,
		};

		template<typename T, sincos_op Op = sincos_op::OP_SINCOS, typename V, typename I = int_of_size_t<sizeof(T)>>
		[[nodiscard]] DPM_FORCEINLINE sincos_ret<V> eval_sincos(V sign_x, V abs_x) noexcept
		{
			/* y = |x| * 4 / Pi */
			auto y = div<T>(abs_x, fill<V>(pio4<T>));

			/* i = isodd(y) ? y + 1 : y */
			auto i = cvtt<I, T>(y);
			i = add<I>(i, fill<decltype(i)>(I{1}));
			i = bit_and(i, fill<decltype(i)>(~I{1}));
			y = cvt<T, I>(i);

			/* Extract sign bit mask */
			const auto bit2 = bit_shiftl<I, sizeof(I) * 8 - 2>(bit_and(i, fill<decltype(i)>(I{2})));
			const auto bit4 = bit_shiftl<I, sizeof(I) * 8 - 3>(bit_and(i, fill<decltype(i)>(I{4})));
			const auto sign_cos = bit_xor(std::bit_cast<V>(bit4), std::bit_cast<V>(bit2));
			const auto sign_sin = bit_xor(std::bit_cast<V>(bit4), sign_x);

			/* Polynomial selection mask (i & 2) */
			const auto p_mask = std::bit_cast<V>(cmp_ne<I>(bit2, setzero<decltype(i)>()));

			auto z = fmadd(y, fill<V>(dp_sincos<T>[0]), abs_x);
			z = fmadd(y, fill<V>(dp_sincos<T>[1]), z);
			z = fmadd(y, fill<V>(dp_sincos<T>[2]), z);
			const auto zz = mul<T>(z, z);

			/* 0 <= a <= Pi/4 : sincof(zz) * zz * z + z */
			const auto p1 = fmadd(mul<T>(polevl(zz, std::span{sincof<T>}), zz), z, z);

			/* Pi/4 <= a <= 0 : coscof(zz) * zz * zz - 0.5 * zz + 1 */
			auto p2 = mul<T>(polevl(zz, std::span{coscof<T>}), mul<T>(zz, zz));
			p2 = add<T>(fmadd(fill<V>(-half<T>), zz, p2), fill<V>(one<T>));

			auto p_sin = undefined<V>(), p_cos = undefined<V>();
			if constexpr (Op & sincos_op::OP_SIN)
			{
				/* Select between p1 and p2 & restore sign */
				p_sin = blendv<T>(p1, p2, p_mask);  /* p_sin = p_mask ? p2 : p1 */
				p_sin = bit_xor(p_sin, sign_sin);   /* p_sin = sign_sin ? -p_sin : p_sin */
			}
			if constexpr (Op & sincos_op::OP_COS)
			{
				/* Select between p1 and p2 & restore sign */
				p_cos = blendv<T>(p2, p1, p_mask);  /* p_cos = p_mask ? p1 : p2 */
				p_cos = bit_xor(p_cos, sign_cos);   /* p_cos = sign_cos ? -p_cos : p_cos */
			}
			return {p_sin, p_cos};
		}

		[[nodiscard]] vec2_return_t<__m128, __m128> DPM_API_PUBLIC DPM_MATHFUNC sincos_f32x4(__m128 x) noexcept;
		DPM_FORCEINLINE void sincos(__m128 x, __m128 &s, __m128 &c) noexcept { s = vec2_call(sincos_f32x4, x, c); }

		[[nodiscard]] vec2_return_t<__m128d, __m128d> DPM_API_PUBLIC DPM_MATHFUNC sincos_f64x2(__m128d x) noexcept;
		DPM_FORCEINLINE void sincos(__m128d x, __m128d &s, __m128d &c) noexcept { s = vec2_call(sincos_f64x2, x, c); }

		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC sin(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC cos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC tan(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC asin(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC acos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC atan(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC atan2(__m128 a, __m128 b) noexcept;

		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC sin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC cos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC tan(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC asin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC acos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC atan(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC atan2(__m128d a, __m128d b) noexcept;

#ifdef DPM_HAS_AVX
		[[nodiscard]] vec2_return_t<__m256, __m256> DPM_API_PUBLIC DPM_MATHFUNC sincos_f32x8(__m256 x) noexcept;
		DPM_FORCEINLINE void sincos(__m256 x, __m256 &s, __m256 &c) noexcept { s = vec2_call(sincos_f32x8, x, c); }

		[[nodiscard]] vec2_return_t<__m256d, __m256d> DPM_API_PUBLIC DPM_MATHFUNC sincos_f64x4(__m256d x) noexcept;
		DPM_FORCEINLINE void sincos(__m256d x, __m256d &s, __m256d &c) noexcept { s = vec2_call(sincos_f64x4, x, c); }

		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC sin(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC cos(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC tan(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC asin(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC acos(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC atan(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC atan2(__m256 a, __m256 b) noexcept;

		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC sin(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC cos(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC tan(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC asin(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC acos(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC atan(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC atan2(__m256d a, __m256d b) noexcept;
#endif
#else
		DPM_FORCEINLINE void sincos(__m128 x, __m128 &s, __m128 &c) noexcept { s = _mm_sincos_ps(c, x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 sin(__m128 x) noexcept { return _mm_sin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 cos(__m128 x) noexcept { return _mm_cos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 tan(__m128 x) noexcept { return _mm_tan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 asin(__m128 x) noexcept { return _mm_asin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 acos(__m128 x) noexcept { return _mm_acos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 atan(__m128 x) noexcept { return _mm_atan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 atan2(__m128 a, __m128 b) noexcept { return _mm_atan2_ps(a, b); }

#ifdef DPM_HAS_SSE2
		DPM_FORCEINLINE void sincos(__m128d x, __m128d &s, __m128d &c) noexcept { s = _mm_sincos_pd(c, x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d sin(__m128d x) noexcept { return _mm_sin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d cos(__m128d x) noexcept { return _mm_cos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d tan(__m128d x) noexcept { return _mm_tan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d asin(__m128d x) noexcept { return _mm_asin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d acos(__m128d x) noexcept { return _mm_acos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d atan(__m128d x) noexcept { return _mm_atan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d atan2(__m128d a, __m128d b) noexcept { return _mm_atan2_pd(a, b); }
#endif
#ifdef DPM_HAS_AVX
		DPM_FORCEINLINE void sincos(__m256 x, __m256 &s, __m256 &c) noexcept { s = _mm256_sincos_ps(c, x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 sin(__m256 x) noexcept { return _mm256_sin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 cos(__m256 x) noexcept { return _mm256_cos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 tan(__m256 x) noexcept { return _mm256_tan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 asin(__m256 x) noexcept { return _mm256_asin_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 acos(__m256 x) noexcept { return _mm256_acos_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 atan(__m256 x) noexcept { return _mm256_atan_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 atan2(__m256 a, __m256 b) noexcept { return _mm256_atan2_ps(a, b); }

		DPM_FORCEINLINE void sincos(__m256d x, __m256d &s, __m256d &c) noexcept { s = _mm256_sincos_pd(c, x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d sin(__m256d x) noexcept { return _mm256_sin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d cos(__m256d x) noexcept { return _mm256_cos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d tan(__m256d x) noexcept { return _mm256_tan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d asin(__m256d x) noexcept { return _mm256_asin_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d acos(__m256d x) noexcept { return _mm256_acos_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d atan(__m256d x) noexcept { return _mm256_atan_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d atan2(__m256d a, __m256d b) noexcept { return _mm256_atan2_pd(a, b); }
#endif
#endif
	}

#if defined(DPM_USE_SVML) || defined(DPM_HAS_SSE2)
	/** Calculates sine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> sin(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(sin(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::sin(x); }, result, x);
			return result;
		}
	}
	/** Calculates cosine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> cos(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(cos(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::cos(x); }, result, x);
			return result;
		}
	}
	/** Calculates tangent of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> tan(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(tan(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::tan(x); }, result, x);
			return result;
		}
	}
	/** Calculates arc-sine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> asin(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(asin(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::asin(x); }, result, x);
			return result;
		}
	}
	/** Calculates arc-cosine of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> acos(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(acos(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::acos(x); }, result, x);
			return result;
		}
	}
	/** Calculates arc-tangent of elements in vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> atan(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(atan(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::atan(x); }, result, x);
			return result;
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates sine and cosine of elements in vector \a x, and assigns results to elements of \a out_sin and \a out_cos respectively. */
		template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		constexpr DPM_FORCEINLINE void sincos(const detail::x86_simd<T, N, A> &x, detail::x86_simd<T, N, A> &out_sin, detail::x86_simd<T, N, A> &out_cos) noexcept
		{
			if (!std::is_constant_evaluated())
				detail::vectorize([](auto x, auto &sin_x, auto &cos_x) { detail::sincos(x, sin_x, cos_x); }, x, out_sin, out_cos);
			else
			{
				const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
				simd<T, simd_abi::ext::packed_buffer<N>> packed_sin = {};
				simd<T, simd_abi::ext::packed_buffer<N>> packed_cos = {};

				sincos(packed_x, packed_sin, packed_cos);
				out_sin = static_cast<detail::x86_simd<T, N, A>>(packed_sin);
				out_cos = static_cast<detail::x86_simd<T, N, A>>(packed_cos);
			}
		}
	}
#endif

#ifdef DPM_USE_SVML
	/** Calculates arc-tangent of quotient of elements in vectors \a a and \a b. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> atan2(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_a = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(a);
			const auto packed_b = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(b);
			return static_cast<detail::x86_simd<T, N, A>>(atan2(packed_a, packed_b));
		}
		else
		{
			detail::x86_simd<T, N, A> result = {};
			detail::vectorize([](auto &res, auto a, auto b) { res = detail::atan2(a, b); }, result, a, b);
			return result;
		}
	}
#endif
}

#endif