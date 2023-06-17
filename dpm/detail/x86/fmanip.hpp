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
		[[nodiscard]] vec2_return_t<__m128, __m128i> DPM_API_PUBLIC DPM_MATHFUNC frexp_f32x4(__m128 x) noexcept;
		[[nodiscard]] vec2_return_t<__m128d, __m128i> DPM_API_PUBLIC DPM_MATHFUNC frexp_f64x2(__m128d x) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m128 frexp(__m128 x, __m128i *out, std::size_t i) noexcept { return vec2_call(frexp_f32x4, x, out[i]); }
		[[nodiscard]] DPM_FORCEINLINE __m128d frexp(__m128d x, __m128i *out, std::size_t i) noexcept
		{
			__m128i tmp;
			x = vec2_call(frexp_f64x2, x, tmp);
			auto *out64 = reinterpret_cast<alias_t<double> *>(out);
			out64[i] = _mm_cvtsd_f64(std::bit_cast<__m128d>(cvt_i64_i32(tmp)));
			return x;
		}
		DPM_FORCEINLINE void frexp2(__m128d &x0, __m128d &x1, __m128i *out, std::size_t i) noexcept
		{
			__m128i tmp0, tmp1;
			x0 = vec2_call(frexp_f64x2, x0, tmp0);
			x1 = vec2_call(frexp_f64x2, x1, tmp1);
			out[i / 2] = cvt_i64x2_i32(tmp0, tmp1);
		}

		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC scalbn(__m128 x, __m128i exp) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC scalbn(__m128d x, __m128i exp) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m128 scalbn32(__m128 x, const __m128i *exp, std::size_t i) noexcept { return scalbn(x, exp[i]); }
		[[nodiscard]] DPM_FORCEINLINE __m128d scalbn32(__m128d x, const __m128i *exp, std::size_t i) noexcept
		{
			auto exp64 = exp[i / 2];
			if (const auto sign = _mm_srai_epi32(exp64, 31); i % 2)
				exp64 = _mm_unpackhi_epi32(exp64, sign);
			else
				exp64 = _mm_unpacklo_epi32(exp64, sign);
			return scalbn(x, exp64);
		}

		[[nodiscard]] DPM_FORCEINLINE __m128 scalbn64(__m128 x, const __m128i *exp, std::size_t i) noexcept
		{
			const auto exph = std::bit_cast<__m128>(exp[i * 2 + 1]);
			const auto expl = std::bit_cast<__m128>(exp[i * 2]);
			return scalbn(x, std::bit_cast<__m128i>(_mm_shuffle_ps(expl, exph, _MM_SHUFFLE(2, 0, 2, 0))));
		}
		[[nodiscard]] DPM_FORCEINLINE __m128d scalbn64(__m128d x, const __m128i *exp, std::size_t i) noexcept { return scalbn(x, exp[i]); }

		[[nodiscard]] vec2_return_t<__m128, __m128> DPM_API_PUBLIC DPM_MATHFUNC modf_f32x4(__m128 x) noexcept;
		[[nodiscard]] vec2_return_t<__m128d, __m128d> DPM_API_PUBLIC DPM_MATHFUNC modf_f64x2(__m128d x) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m128 modf(__m128 x, __m128 &i) noexcept { return vec2_call(modf_f32x4, x, i); }
		[[nodiscard]] DPM_FORCEINLINE __m128d modf(__m128d x, __m128d &i) noexcept { return vec2_call(modf_f64x2, x, i);}

		[[nodiscard]] __m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128 x) noexcept;
		[[nodiscard]] __m128i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m128d x) noexcept;

		DPM_FORCEINLINE void ilogb(__m128 x, __m128i *out, std::size_t i) noexcept { out[i] = ilogb(x); }
		DPM_FORCEINLINE void ilogb(__m128d x, __m128i *out, std::size_t i) noexcept
		{
			auto *out64 = reinterpret_cast<alias_t<std::int64_t> *>(out);
			_mm_storeu_si64(out64 + i, cvt_i64_i32(ilogb(x)));
		}
		DPM_FORCEINLINE void ilogb2(__m128d x0, __m128d x1, __m128i *out, std::size_t i) noexcept
		{
			const auto i0 = ilogb(x0);
			const auto i1 = ilogb(x1);
			out[i / 2] = cvt_i64x2_i32(i0, i1);
		}

#ifndef DPM_USE_SVML
		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC logb(__m128 x) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC logb(__m128d x) noexcept;
#endif

		[[nodiscard]] __m128 DPM_API_PUBLIC DPM_MATHFUNC nextafter(__m128 from, __m128 to) noexcept;
		[[nodiscard]] __m128d DPM_API_PUBLIC DPM_MATHFUNC nextafter(__m128d from, __m128d to) noexcept;
#endif

#ifdef DPM_HAS_AVX
		[[nodiscard]] vec2_return_t<__m256, __m256i> DPM_API_PUBLIC DPM_MATHFUNC frexp_f32x8(__m256 x) noexcept;
		[[nodiscard]] vec2_return_t<__m256d, __m256i> DPM_API_PUBLIC DPM_MATHFUNC frexp_f64x4(__m256d x) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m256 frexp(__m256 x, __m256i *out, std::size_t i) noexcept { return vec2_call(frexp_f32x8, x, out[i]); }
		[[nodiscard]] DPM_FORCEINLINE __m256d frexp(__m256d x, __m128i *out, std::size_t i) noexcept
		{
			__m256i tmp;
			x = vec2_call(frexp_f64x4, x, tmp);
			out[i] = cvt_i64_i32(tmp);
			return x;
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d frexp(__m256d x, __m256i *out, std::size_t i) noexcept { return frexp(x, reinterpret_cast<__m128i *>(out), i); }
		DPM_FORCEINLINE void frexp2(__m256d &x0, __m256d &x1, __m256i *out, std::size_t i) noexcept
		{
			__m256i tmp0, tmp1;
			x0 = vec2_call(frexp_f64x4, x0, tmp0);
			x1 = vec2_call(frexp_f64x4, x1, tmp1);
			out[i / 2] = cvt_i64x2_i32(tmp0, tmp1);
		}

		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC scalbn(__m256 x, __m256i exp) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC scalbn(__m256d x, __m256i exp) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m256 scalbn32(__m256 x, const __m256i *exp, std::size_t i) noexcept { return scalbn(x, exp[i]); }
		[[nodiscard]] DPM_FORCEINLINE __m256d scalbn32(__m256d x, const __m128i *exp, std::size_t i) noexcept
		{
			const auto exph = exp[i + 1];
			const auto expl = exp[i];
			const auto xh = _mm256_castpd256_pd128(_mm256_permute2f128_pd(x, x, 1));
			const auto xl = _mm256_castpd256_pd128(x);
			const auto h = scalbn(xh, _mm_unpackhi_epi32(exph, _mm_srai_epi32(exph, 31)));
			const auto l = scalbn(xl, _mm_unpackhi_epi32(expl, _mm_srai_epi32(expl, 31)));
			return _mm256_set_m128d(h, l);
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d scalbn32(__m256d x, const __m256i *exp, std::size_t i) noexcept
		{
#ifdef DPM_HAS_AVX2
			auto exp64 = exp[i / 2];
			if (const auto sign = _mm256_srai_epi32(exp64, 31); i % 2)
				exp64 = _mm256_unpackhi_epi32(exp64, sign);
			else
				exp64 = _mm256_unpacklo_epi32(exp64, sign);
			return scalbn(x, exp64);
#else
			return scalbn32(x, reinterpret_cast<const __m128i *>(exp), i);
#endif
		}

		[[nodiscard]] DPM_FORCEINLINE __m128 scalbn64(__m128 x, const __m256i *exp, std::size_t i) noexcept{return scalbn(x, cvt_i64_i32(exp[i]));}
		[[nodiscard]] DPM_FORCEINLINE __m256 scalbn64(__m256 x, const __m256i *exp, std::size_t i) noexcept
		{
			const auto exph = cvt_i64_i32(exp[i]);
			const auto expl = cvt_i64_i32(exp[i]);
			return scalbn(x, _mm256_set_m128i(exph, expl));
		}
		[[nodiscard]] DPM_FORCEINLINE __m256d scalbn64(__m256d x, const __m256i *exp, std::size_t i) noexcept { return scalbn(x, exp[i]); }

		[[nodiscard]] vec2_return_t<__m256, __m256> DPM_API_PUBLIC DPM_MATHFUNC modf_f32x8(__m256 x) noexcept;
		[[nodiscard]] vec2_return_t<__m256d, __m256d> DPM_API_PUBLIC DPM_MATHFUNC modf_f64x4(__m256d x) noexcept;

		[[nodiscard]] DPM_FORCEINLINE __m256 modf(__m256 x, __m256 &i) noexcept { return vec2_call(modf_f32x8, x, i); }
		[[nodiscard]] DPM_FORCEINLINE __m256d modf(__m256d x, __m256d &i) noexcept { return vec2_call(modf_f64x4, x, i);}

		[[nodiscard]] __m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256 x) noexcept;
		[[nodiscard]] __m256i DPM_API_PUBLIC DPM_MATHFUNC ilogb(__m256d x) noexcept;

		DPM_FORCEINLINE void ilogb(__m256 x, __m256i *out, std::size_t i) noexcept { out[i] = ilogb(x); }
		DPM_FORCEINLINE void ilogb(__m256d x, __m128i *out, std::size_t i) noexcept { out[i] = cvt_i64_i32(ilogb(x)); }
		DPM_FORCEINLINE void ilogb(__m256d x, __m256i *out, std::size_t i) noexcept { ilogb(x, reinterpret_cast<__m128i *>(out), i); }
		DPM_FORCEINLINE void ilogb2(__m256d x0, __m256d x1, __m256i *out, std::size_t i) noexcept
		{
			const auto i0 = ilogb(x0);
			const auto i1 = ilogb(x1);
			out[i / 2] = cvt_i64x2_i32(i0, i1);
		}

#ifndef DPM_USE_SVML
		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC logb(__m256 x) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC logb(__m256d x) noexcept;
#endif

		[[nodiscard]] __m256 DPM_API_PUBLIC DPM_MATHFUNC nextafter(__m256 from, __m256 to) noexcept;
		[[nodiscard]] __m256d DPM_API_PUBLIC DPM_MATHFUNC nextafter(__m256d from, __m256d to) noexcept;
#endif
	}

#ifdef DPM_HAS_SSE2
	/** Decomposes elements of vector \a x into a normalized fraction and a power-of-two exponent, stores the exponent in \a exp, and returns the fraction. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> frexp(const detail::x86_simd<T, N, A> &x, detail::x86_simd<int, N, A> *exp) noexcept
	{
		if (std::is_constant_evaluated())
		{
			simd<int, simd_abi::ext::packed_buffer<N>> packed_exp = {};
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			const auto result = static_cast<detail::x86_simd<T, N, A>>(frexp(packed_x, &packed_exp));
			*exp = static_cast<detail::x86_simd<int, N, A>>(packed_exp);
			return result;
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			auto result_data = ext::to_native_data(result);
			const auto x_data = ext::to_native_data(x);
			auto exp_data = ext::to_native_data(*exp);

			std::size_t i = 0;
			constexpr auto native_size = ext::native_data_size_v < detail::x86_simd < T, N, A>>;
			if constexpr (sizeof(T) > sizeof(int) && ext::native_data_size_v < detail::x86_simd<int, N, A>> < native_size)
				for (; i + 1 < native_size; i += 2) detail::frexp2((result_data[i] = x_data[i]), (result_data[i + 1] = x_data[i + 1]), exp_data.data(), i);
			for (; i < native_size; ++i) result_data[i] = detail::frexp(x_data[i], exp_data.data(), i);

			return result;
		}
	}

	/** Multiplies elements of vector \a x by `2` raised to power specified by elements of vector \a exp. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> ldexp(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<int, N, A> &exp) noexcept { return scalbn(x, exp); }
	/** @copydoc ldexp */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> scalbn(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<int, N, A> &exp) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_exp = static_cast<simd<int, simd_abi::ext::packed_buffer<N>>>(exp);
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(scalbn(packed_x, packed_exp));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			auto result_data = ext::to_native_data(result);
			const auto exp_data = ext::to_native_data(exp);
			const auto x_data = ext::to_native_data(x);
			for (std::size_t i = 0; i < ext::native_data_size_v < detail::x86_simd < T, N, A >>; ++i)
				result_data[i] = detail::scalbn32(x_data[i], exp_data.data(), i);
			return result;
		}
	}
	/** @copydoc ldexp */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<long, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> scalbln(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<long, N, A> &exp) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_exp = static_cast<simd<long, simd_abi::ext::packed_buffer<N>>>(exp);
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(scalbln(packed_x, packed_exp));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			auto result_data = ext::to_native_data(result);
			const auto exp_data = ext::to_native_data(exp);
			const auto x_data = ext::to_native_data(x);
			for (std::size_t i = 0; i < ext::native_data_size_v < detail::x86_simd < T, N, A >>; ++i)
			{
				if constexpr (sizeof(long) == sizeof(std::int64_t))
					result_data[i] = detail::scalbn64(x_data[i], exp_data.data(), i);
				else
					result_data[i] = detail::scalbn32(x_data[i], exp_data.data(), i);
			}
			return result;
		}
	}

	/** Decomposes elements of vector \a x into integral and fractional parts, returning the fractional and storing the integral in \a out_i. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> modf(const detail::x86_simd<T, N, A> &x, detail::x86_simd<T, N, A> *out_i) noexcept
	{
		if (std::is_constant_evaluated())
		{
			simd<T, simd_abi::ext::packed_buffer<N>> packed_i = {};
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);

			const auto result = static_cast<detail::x86_simd<T, N, A>>(modf(packed_x, &packed_i));
			*out_i = static_cast<detail::x86_simd<T, N, A>>(packed_i);
			return result;
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x, auto &i) { res = detail::modf(x, i); }, result, x, *out_i);
			return result;
		}
	}
	/** Extracts unbiased exponent of elements in vector \a x as integers. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<int, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<int, N, A> ilogb(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<int, N, A>>(ilogb(packed_x));
		}
		else
		{
			detail::x86_simd<int, N, A> result = {};
			auto result_data = ext::to_native_data(result);
			const auto x_data = ext::to_native_data(x);

			std::size_t i = 0;
			constexpr auto native_size = ext::native_data_size_v < detail::x86_simd < T, N, A>>;
			if constexpr (sizeof(T) > sizeof(int) && ext::native_data_size_v < detail::x86_simd < int, N, A >> < native_size)
				for (; i + 1 < native_size; i += 2) detail::ilogb2(x_data[i], x_data[i + 1], result_data.data(), i);
			for (; i < native_size; ++i) detail::ilogb(x_data[i], result_data.data(), i);

			return result;
		}
	}
	/** Finds next representable value from elements of vector \a from to elements of vector \a to. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> nextafter(const detail::x86_simd<T, N, A> &from, const detail::x86_simd<T, N, A> &to) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_from = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(from);
			const auto packed_to = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(to);
			return static_cast<detail::x86_simd<T, N, A>>(nextafter(packed_from, packed_to));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto from, auto to) { res = detail::nextafter(from, to); }, result, from, to);
			return result;
		}
	}

	/** Multiplies elements of vector \a x by `2` raised to power \a exp. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> ldexp(const detail::x86_simd<T, N, A> &x, int exp) noexcept { return scalbn(x, exp); }
	/** @copydoc ldexp */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> scalbn(const detail::x86_simd<T, N, A> &x, int exp) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(scalbn(packed, exp));
		}
		else
		{
			using exp_t = int_of_size_t<sizeof(T)>;
			using exp_vector = detail::select_vector_t<exp_t, sizeof(ext::native_data_type_t < detail::x86_simd < T, N, A >>)>;

			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([exp = detail::fill<exp_vector>(static_cast<exp_t>(exp))](auto &res, auto x) { res = detail::scalbn(x, exp); }, result, x);
			return result;
		}
	}
	/** @copydoc ldexp */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> scalbln(const detail::x86_simd<T, N, A> &x, long exp) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(scalbln(packed, exp));
		}
		else
		{
			using exp_t = int_of_size_t<sizeof(T)>;
			using exp_vector = detail::select_vector_t<exp_t, sizeof(ext::native_data_type_t < detail::x86_simd < T, N, A >>)>;

			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([exp = detail::fill<exp_vector>(static_cast<exp_t>(exp))](auto &res, auto x) { res = detail::scalbn(x, exp); }, result, x);
			return result;
		}
	}
#endif

#if defined(DPM_USE_SVML) || defined(DPM_HAS_SSE2)
	/** Extracts unbiased exponent of elements in vector \a x as floats. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE detail::x86_simd<T, N, A> logb(const detail::x86_simd<T, N, A> &x) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(logb(packed));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x) { res = detail::logb(x); }, result, x);
			return result;
		}
	}
#endif

	/** Copies sign bit from elements of vector \a sign to elements of vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &sign) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed_sign = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			const auto packed_x = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(copysign(packed_x, packed_sign));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			detail::vectorize([](auto &res, auto x, auto s) { res = detail::copysign<T>(x, detail::masksign<T>(s)); }, result, x, sign);
			return result;
		}
	}
	/** Copies sign bit from \a sign to elements of vector \a x. */
	template<std::floating_point T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
	[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, T sign) noexcept
	{
		if (std::is_constant_evaluated())
		{
			const auto packed = static_cast<simd<T, simd_abi::ext::packed_buffer<N>>>(x);
			return static_cast<detail::x86_simd<T, N, A>>(copysign(packed, sign));
		}
		else
		{
			detail::x86_simd <T, N, A> result = {};
			const auto sign_vec = detail::masksign<T>(detail::fill < ext::native_data_type_t < detail::x86_simd < T, N, A>>>(sign));
			detail::vectorize([s = sign_vec](auto &res, auto x) { res = detail::copysign<T>(x, s); }, result, x);
			return result;
		}
	}
}