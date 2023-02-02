/*
 * Created by switchblade on 2023-02-01.
 */

#include "mbase.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include "../dispatch.hpp"
#include "cpuid.hpp"
#include "class.hpp"
#include "sign.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
#if defined(DPM_HAS_AVX2) || defined(DPM_DYNAMIC_DISPATCH)
	inline static __m128d DPM_MATHFUNC("avx2") fmod_avx2(__m128d sign_a, __m128d abs_a, __m128d abs_b, __m128d zero_mask, __m128d fwd_mask, [[maybe_unused]] __m128d nan_mask) noexcept
	{
		const auto max_subnorm = _mm_set1_epi64x(0x000fffffffffffff);
		const auto min_norm = _mm_set1_epi64x(0x0010000000000000);
		const auto izero = setzero<__m128i>();
		const auto fzero = setzero<__m128d>();
		const auto one = _mm_set1_epi64x(1);
		auto ha = std::bit_cast<__m128i>(abs_a);
		auto hb = std::bit_cast<__m128i>(abs_b);

		/* Find exponent of and normalize `a` */
		auto ia = _mm_sub_epi64(_mm_srli_epi64(ha, 52), _mm_set1_epi64x(1023));
		auto a_sub = _mm_cmpgt_epi64(min_norm, ha);
		if (_mm_movemask_epi8(a_sub)) [[unlikely]]
		{
			auto sub_i = _mm_set1_epi64x(-1022), i = _mm_slli_epi64(ha, 11);
			for (; _mm_movemask_epi8(i); i = _mm_slli_epi64(i, 1)) _mm_sub_epi64(sub_i, one);
			ia = _mm_blendv_epi8(ia, sub_i, a_sub);
		}

		/* Find exponent of and normalize `b` */
		auto ib = _mm_sub_epi64(_mm_srli_epi64(hb, 52), _mm_set1_epi64x(1023));
		const auto b_sub = _mm_cmpgt_epi64(min_norm, hb);
		if (_mm_movemask_epi8(b_sub)) [[unlikely]]
		{
			auto sub_i = _mm_set1_epi64x(-1022), i = _mm_slli_epi64(hb, 11);
			for (; _mm_movemask_epi8(i); i = _mm_slli_epi64(i, 1)) _mm_sub_epi64(sub_i, one);
			ib = _mm_blendv_epi8(ib, sub_i, b_sub);
		}

		/* Align a with b */
		const auto norm_ha = _mm_or_si128(min_norm, _mm_and_si128(max_subnorm, ha));
		const auto norm_hb = _mm_or_si128(min_norm, _mm_and_si128(max_subnorm, hb));
		const auto sub_ha = _mm_sllv_epi64(ha, _mm_sub_epi64(_mm_set1_epi64x(-1022), ia));
		const auto sub_hb = _mm_sllv_epi64(hb, _mm_sub_epi64(_mm_set1_epi64x(-1022), ib));
		ha = _mm_blendv_epi8(norm_ha, sub_ha, a_sub);
		hb = _mm_blendv_epi8(norm_hb, sub_hb, b_sub);

		/* Fixed point fmod. */
		for (auto n = _mm_sub_epi64(ia, ib); _mm_movemask_epi8(n); n = _mm_sub_epi64(n, one))
		{
			const auto hc = _mm_sub_epi64(ha, hb);
			if (!_mm_movemask_epi8(hc)) [[unlikely]] /* Bail if ha - hb == 0. */
				return _mm_or_pd(sign_a, fzero);

			/* hc < 0 ? ha * 2 : hc * 2 */
			const auto neg = _mm_cmpgt_epi64(izero, hc);
			const auto ha2 = _mm_add_epi64(ha, ha);
			const auto hc2 = _mm_add_epi64(hc, hc);
			ha = _mm_blendv_epi8(ha2, hc2, neg);
			zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi64(hc, izero));
		}

		const auto hc = _mm_sub_epi64(ha, hb);
		const auto neg = _mm_cmpgt_epi64(izero, hc);
		ha = _mm_blendv_epi8(hc, ha, neg);

		/* Normalize ha. */
		for (;;)
		{
			a_sub = _mm_cmpgt_epi64(min_norm, ha);
			if (_mm_movemask_epi8(a_sub)) [[unlikely]]
				break;

			/* Repeat ib -= 1; ha *= 2; until all elements of ha are normal. */
			ib = _mm_blendv_pd(ib, _mm_sub_epi64(ib, one), a_sub);
			ha = _mm_blendv_pd(ha, _mm_add_epi64(ha, ha), a_sub);
		}

		/* Normalize result. */
		const auto x_sub = _mm_cmpgt_epi64(ib, _mm_set1_epi64x(-1021));
		/* Normal result */
		auto xn = _mm_add_epi64(ib, _mm_set1_epi64x(1023));
		xn = _mm_or_si128(_mm_slli_epi64(xn, 52), _mm_sub_epi64(hb, min_norm));
		/* Subnormal result */
		const auto xs = _mm_srlv_epi64(ha, _mm_sub_epi64(_mm_set1_epi64x(-1022), ib));
		auto x = std::bit_cast<__m128d>(_mm_blendv_epi8(xs, xn, x_sub));

		/* Mask zero results. */
		x = _mm_blendv_pd(x, fzero, zero_mask);
		/* Mask forwarded results. */
		x = _mm_blendv_pd(x, abs_a, fwd_mask);
		/* Mask NaNs. */
#if defined(DPM_PROPAGATE_NAN) || defined(DPM_HANDLE_ERRORS)
		const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
		x = _mm_blendv_pd(x, nan, nan_mask);
#endif
		/* Restore sign & return. */
		return _mm_or_pd(x, sign_a);
	}

#ifdef DPM_HAS_AVX
	inline static __m256d DPM_MATHFUNC("avx2") fmod_avx2(__m256d sign_a, __m256d abs_a, __m256d abs_b, __m256d zero_mask, __m256d fwd_mask, [[maybe_unused]] __m256d nan_mask) noexcept;
#endif
#endif

#ifdef DPM_HAS_AVX
	inline static __m256d DPM_MATHFUNC("avx") fmod_avx(__m256d sign_a, __m256d abs_a, __m256d abs_b, __m256d zero_mask, __m256d fwd_mask, [[maybe_unused]] __m256d nan_mask) noexcept;
#endif

#if defined(DPM_HAS_SSE4_1) || defined(DPM_DYNAMIC_DISPATCH)
	inline static __m128d DPM_MATHFUNC("sse4.1") fmod_sse4_1(__m128d sign_a, __m128d abs_a, __m128d abs_b, __m128d zero_mask, __m128d fwd_mask, [[maybe_unused]] __m128d nan_mask) noexcept
	{
		const auto max_subnorm = _mm_set1_epi64x(0x000fffffffffffff);
		const auto min_norm = _mm_set1_epi64x(0x0010000000000000);
		const auto izero = setzero<__m128i>();
		const auto fzero = setzero<__m128d>();
		const auto one = _mm_set1_epi64x(1);
		auto ha = std::bit_cast<__m128i>(abs_a);
		auto hb = std::bit_cast<__m128i>(abs_b);

		/* Find exponent of and normalize `a` */
		auto ia = _mm_sub_epi64(_mm_srli_epi64(ha, 52), _mm_set1_epi64x(1023));
		auto a_sub = _mm_cmpgt_epi64(min_norm, ha);
		if (_mm_movemask_epi8(a_sub)) [[unlikely]]
		{
			auto sub_i = _mm_set1_epi64x(-1022), i = _mm_slli_epi64(ha, 11);
			for (; _mm_movemask_epi8(i); i = _mm_slli_epi64(i, 1)) _mm_sub_epi64(sub_i, one);
			ia = _mm_blendv_epi8(ia, sub_i, a_sub);
		}

		/* Find exponent of and normalize `b` */
		auto ib = _mm_sub_epi64(_mm_srli_epi64(hb, 52), _mm_set1_epi64x(1023));
		const auto b_sub = _mm_cmpgt_epi64(min_norm, hb);
		if (_mm_movemask_epi8(b_sub)) [[unlikely]]
		{
			auto sub_i = _mm_set1_epi64x(-1022), i = _mm_slli_epi64(hb, 11);
			for (; _mm_movemask_epi8(i); i = _mm_slli_epi64(i, 1)) _mm_sub_epi64(sub_i, one);
			ib = _mm_blendv_epi8(ib, sub_i, b_sub);
		}

		/* Align a with b */
		const auto norm_ha = _mm_or_si128(min_norm, _mm_and_si128(max_subnorm, ha));
		const auto norm_hb = _mm_or_si128(min_norm, _mm_and_si128(max_subnorm, hb));
		const auto sub_ha = bit_shiftl64_sse(ha, _mm_sub_epi64(_mm_set1_epi64x(-1022), ia));
		const auto sub_hb = bit_shiftl64_sse(hb, _mm_sub_epi64(_mm_set1_epi64x(-1022), ib));
		ha = _mm_blendv_epi8(norm_ha, sub_ha, a_sub);
		hb = _mm_blendv_epi8(norm_hb, sub_hb, b_sub);

		/* Fixed point fmod. */
		for (auto n = _mm_sub_epi64(ia, ib); _mm_movemask_epi8(n); n = _mm_sub_epi64(n, one))
		{
			const auto hc = _mm_sub_epi64(ha, hb);
			if (!_mm_movemask_epi8(hc)) [[unlikely]] /* Bail if ha - hb == 0. */
				return _mm_or_pd(sign_a, fzero);

			/* hc < 0 ? ha * 2 : hc * 2 */
			const auto neg = _mm_cmpgt_epi64(izero, hc);
			const auto ha2 = _mm_add_epi64(ha, ha);
			const auto hc2 = _mm_add_epi64(hc, hc);
			ha = _mm_blendv_epi8(ha2, hc2, neg);
			zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi64(hc, izero));
		}

		const auto hc = _mm_sub_epi64(ha, hb);
		const auto neg = _mm_cmpgt_epi64(izero, hc);
		ha = _mm_blendv_epi8(hc, ha, neg);

		/* Normalize ha. */
		for (;;)
		{
			a_sub = _mm_cmpgt_epi64(min_norm, ha);
			if (_mm_movemask_epi8(a_sub)) [[unlikely]]
				break;

			/* Repeat ib -= 1; ha *= 2; until all elements of ha are normal. */
			ib = _mm_blendv_pd(ib, _mm_sub_epi64(ib, one), a_sub);
			ha = _mm_blendv_pd(ha, _mm_add_epi64(ha, ha), a_sub);
		}

		/* Normalize result. */
		const auto x_sub = _mm_cmpgt_epi64(ib, _mm_set1_epi64x(-1021));
		/* Normal result */
		auto xn = _mm_add_epi64(ib, _mm_set1_epi64x(1023));
		xn = _mm_or_si128(_mm_slli_epi64(xn, 52), _mm_sub_epi64(hb, min_norm));
		/* Subnormal result */
		const auto xs = bit_shiftr64_sse(ha, _mm_sub_epi64(_mm_set1_epi64x(-1022), ib));
		auto x = std::bit_cast<__m128d>(_mm_blendv_epi8(xs, xn, x_sub));

		/* Mask zero results. */
		x = _mm_blendv_pd(x, fzero, zero_mask);
		/* Mask forwarded results. */
		x = _mm_blendv_pd(x, abs_a, fwd_mask);
		/* Mask NaNs. */
#if defined(DPM_PROPAGATE_NAN) || defined(DPM_HANDLE_ERRORS)
		const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
		x = _mm_blendv_pd(x, nan, nan_mask);
#endif
		/* Restore sign & return. */
		return _mm_or_pd(x, sign_a);
	}
#endif

	inline static __m128d DPM_MATHFUNC("sse2") fmod_sse(__m128d sign_a, __m128d abs_a, __m128d abs_b, __m128d zero_mask, __m128d fwd_mask, [[maybe_unused]] __m128d nan_mask) noexcept
	{
		const auto max_subnorm = _mm_set1_epi64x(0x000fffffffffffff);
		const auto min_norm = _mm_set1_epi64x(0x0010000000000000);
		const auto izero = setzero<__m128i>();
		const auto fzero = setzero<__m128d>();
		const auto one = _mm_set1_epi64x(1);
		auto ha = std::bit_cast<__m128i>(abs_a);
		auto hb = std::bit_cast<__m128i>(abs_b);

		/* Find exponent of and normalize `a` */
		auto ia = _mm_sub_epi64(_mm_srli_epi64(ha, 52), _mm_set1_epi64x(1023));
		auto a_sub = _mm_cmplt_epi32(ha, min_norm);
		a_sub = _mm_or_si128(a_sub, _mm_srli_epi64(a_sub, 32));
		if (_mm_movemask_epi8(a_sub)) [[unlikely]]
		{
			auto sub_i = _mm_set1_epi64x(-1022), i = _mm_slli_epi64(ha, 11);
			for (; _mm_movemask_epi8(i); i = _mm_slli_epi64(i, 1)) _mm_sub_epi64(sub_i, one);
			ia = _mm_or_si128(_mm_andnot_si128(a_sub, ia), _mm_and_si128(a_sub, sub_i));
		}

		/* Find exponent of and normalize `b` */
		auto ib = _mm_sub_epi64(_mm_srli_epi64(hb, 52), _mm_set1_epi64x(1023));
		auto b_sub = _mm_cmplt_epi32(hb, min_norm);
		b_sub = _mm_or_si128(b_sub, _mm_srli_epi64(b_sub, 32));
		if (_mm_movemask_epi8(b_sub)) [[unlikely]]
		{
			auto sub_i = _mm_set1_epi64x(-1022), i = _mm_slli_epi64(hb, 11);
			for (; _mm_movemask_epi8(i); i = _mm_slli_epi64(i, 1)) _mm_sub_epi64(sub_i, one);
			ib = _mm_or_si128(_mm_andnot_si128(b_sub, ib), _mm_and_si128(b_sub, sub_i));
		}

		/* Align a with b */
		const auto norm_ha = _mm_or_si128(min_norm, _mm_and_si128(max_subnorm, ha));
		const auto norm_hb = _mm_or_si128(min_norm, _mm_and_si128(max_subnorm, hb));
		const auto sub_ha = bit_shiftl64_sse(ha, _mm_sub_epi64(_mm_set1_epi64x(-1022), ia));
		const auto sub_hb = bit_shiftl64_sse(hb, _mm_sub_epi64(_mm_set1_epi64x(-1022), ib));
		ha = _mm_or_si128(_mm_andnot_si128(a_sub, norm_ha), _mm_and_si128(a_sub, sub_ha));
		hb = _mm_or_si128(_mm_andnot_si128(b_sub, norm_hb), _mm_and_si128(b_sub, sub_hb));

		/* Fixed point fmod. */
		for (auto n = _mm_sub_epi64(ia, ib); _mm_movemask_epi8(n); n = _mm_sub_epi64(n, one))
		{
			const auto hc = _mm_sub_epi64(ha, hb);
			if (!_mm_movemask_epi8(hc)) [[unlikely]] /* Bail if ha - hb == 0. */
				return _mm_or_pd(sign_a, fzero);

			/* hc < 0 ? ha * 2 : hc * 2 */
			auto neg = _mm_cmpgt_epi32(izero, hc);
			neg = _mm_or_si128(_mm_srli_epi64(neg, 32), neg);
			const auto ha2 = _mm_add_epi64(ha, ha);
			const auto hc2 = _mm_add_epi64(hc, hc);
			ha = _mm_or_si128(_mm_andnot_si128(neg, ha2), _mm_and_si128(neg, hc2));
			zero_mask = _mm_or_si128(zero_mask, _mm_cmpeq_epi64(hc, izero));
		}
		const auto hc = _mm_sub_epi64(ha, hb);
		auto neg = _mm_cmpgt_epi32(izero, hc);
		neg = _mm_or_si128(_mm_srli_epi64(neg, 32), neg);
		ha = _mm_or_si128(_mm_andnot_si128(neg, hc), _mm_and_si128(neg, ha));

		/* Normalize ha. */
		for (;;)
		{
			a_sub = _mm_cmpgt_epi64(min_norm, ha);
			if (_mm_movemask_epi8(a_sub)) [[unlikely]]
				break;

			/* Repeat ib -= 1; ha *= 2; until all elements of ha are normal. */
			ib = _mm_or_si128(_mm_andnot_si128(a_sub, ib), _mm_and_si128(a_sub, _mm_sub_epi64(ib, one)));
			ha = _mm_or_si128(_mm_andnot_si128(a_sub, ha), _mm_and_si128(a_sub, _mm_add_epi64(ha, ha)));
		}

		/* Normalize result. */
		const auto x_sub = _mm_cmpgt_epi64(ib, _mm_set1_epi64x(-1021));
		/* Normal result */
		auto xn = _mm_add_epi64(ib, _mm_set1_epi64x(1023));
		xn = _mm_or_si128(_mm_slli_epi64(xn, 52), _mm_sub_epi64(hb, min_norm));
		/* Subnormal result */
		const auto xs = bit_shiftr64_sse(ha, _mm_sub_epi64(_mm_set1_epi64x(-1022), ib));
		auto x = std::bit_cast<__m128d>(_mm_or_si128(_mm_andnot_si128(x_sub, xs), _mm_and_si128(x_sub, xn)));

		/* Mask zero results. */
		x = _mm_or_pd(_mm_andnot_pd(zero_mask, x), _mm_and_pd(zero_mask, fzero));
		/* Mask forwarded results. */
		x = _mm_or_pd(_mm_andnot_pd(fwd_mask, x), _mm_and_pd(fwd_mask, abs_a));
		/* Mask NaNs. */
#if defined(DPM_PROPAGATE_NAN) || defined(DPM_HANDLE_ERRORS)
		const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
		x = _mm_or_pd(_mm_andnot_pd(nan_mask, x), _mm_and_pd(nan_mask, nan));
#endif
		/* Restore sign & return. */
		return _mm_or_pd(x, sign_a);
	}

	template<typename T, typename V>
	DPM_FORCEINLINE V impl_fmod(V a, V b) noexcept
	{
		constexpr auto extent_bits = movemask_bits_v<T> * sizeof(V) / sizeof(T);
		const auto sign_a = masksign(a);
		const auto abs_a = bit_xor(a, sign_a);
		const auto abs_b = abs(b);

		/* Enforce domain & propagate NaN. */
#if defined(DPM_PROPAGATE_NAN) || defined(DPM_HANDLE_ERRORS)
		auto nan_mask = bit_or(isnan(a), isnan(b));
#ifdef DPM_HANDLE_ERRORS
		const auto inf = fill<V>(std::numeric_limits<T>::infinity());
		const auto b_zero = cmp_eq<T>(abs_b, setzero<V>());
		const auto a_inf = cmp_eq<T>(abs_a, inf);
		const auto b_inf = cmp_eq<T>(abs_b, inf);
		const auto dom_mask = bit_or(b_zero, bit_or(a_inf, b_inf));
		if (movemask<T>(dom_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
		nan_mask = bit_or(nan_mask, dom_mask);
#endif
		/* If either input is NaN or domain error occurs, return NaN. */
		if (movemask<T>(nan_mask) == fill_bits<extent_bits>()) [[unlikely]]
			return fill<V>(std::numeric_limits<T>::quiet_NaN());
#else
		const auto nan_mask = undefined<V>();
#endif

		/* Handle cases when x / y < 0 */
		auto zero_mask = cmp_eq<T>(abs_a, abs_b); /* return 0 if |a| == |b| */
		if (movemask<T>(zero_mask) == fill_bits<extent_bits>()) [[unlikely]]
			return bit_or(setzero<V>(), sign_a);
		auto fwd_mask = cmp_le<T>(abs_a, abs_b);  /* return a if |a| < |b| */
		if (movemask<T>(fwd_mask) == fill_bits<extent_bits>()) [[unlikely]]
			return a;

#if !defined(DPM_HAS_AVX2) && defined(DPM_DYNAMIC_DISPATCH)
		constinit static dispatcher sincos_disp = []() -> V (*)(V, V, V, V, V, V)
		{
			if (cpuid::has_avx2())
				return fmod_avx2;
			else if constexpr(sizeof(V) == 32)
				return fmod_avx;
			else
			{
#ifndef DPM_HAS_SSE4_1
				if (!cpuid::has_sse4_1()) return fmod_sse;
#endif
				return fmod_sse4_1;
			}
		};
		return sincos_disp(sign_a, abs_a, abs_b, zero_mask, fwd_mask, nan_mask);
#elif defined(DPM_HAS_AVX2)
		return fmod_avx2(sign_a, abs_a, abs_b, zero_mask, fwd_mask, nan_mask);
#else
		if constexpr (sizeof(V) == 16)
		{
#ifdef DPM_HAS_SSE4_1
			return fmod_sse4_1(sign_a, abs_a, abs_b, zero_mask, fwd_mask, nan_mask);
#else
			return fmod_sse(sign_a, abs_a, abs_b, zero_mask, fwd_mask, nan_mask);
#endif
		}
		else if constexpr (sizeof(V) == 32)
		{
#ifdef DPM_HAS_SSE4_1
			return fmod_avx2(sign_a, abs_a, abs_b, zero_mask, fwd_mask, nan_mask);
#else
			return fmod_avx(sign_a, abs_a, abs_b, zero_mask, fwd_mask, nan_mask);
#endif
		}
#endif
	}

	//__m128 DPM_PUBLIC DPM_MATHFUNC("sse2") fmod(__m128 a, __m128 b) noexcept { return impl_fmod<float>(a, b); }
	__m128d DPM_PUBLIC DPM_MATHFUNC("sse2") fmod(__m128d a, __m128d b) noexcept { return impl_fmod<double>(a, b); }

#ifdef DPM_HAS_AVX
	//__m256 DPM_PUBLIC DPM_MATHFUNC("avx") fmod(__m256 a, __m256 b) noexcept { return impl_fmod<float>(a, b); }
	__m256d DPM_PUBLIC DPM_MATHFUNC("avx") fmod(__m256d a, __m256d b) noexcept { return impl_fmod<double>(a, b); }
#endif
}

#endif
