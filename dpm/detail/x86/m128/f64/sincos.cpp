/*
 * Created by switchblade on 2023-01-10.
 */

#include "../../../ieee754/const_f64.hpp"
#include "../../../dispatch.hpp"
#include "../../cpuid.hpp"
#include "utility.hpp"
#include "polevl.hpp"
#include "sincos.hpp"
#include "fmadd.hpp"
#include "class.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#ifdef DPM_HANDLE_ERRORS
#ifndef DPM_USE_IMPORT

#include <cfenv>

#endif
#pragma STDC FENV_ACCESS ON
#endif

#define SINCOS_FUNC_CALL(f) f(x, abs_x, DPM_PROPAGATE_NAN_OPT_OR(nan_mask, __m128d{}), DPM_HANDLE_ERRORS_OPT_OR(zero_mask, __m128d{}))
#define SINCOS_FUNC_SIGN(f) f(__m128d x, __m128d abs_x, [[maybe_unused]] __m128d nan_mask, [[maybe_unused]] __m128d zero_mask)

#define FMA_FUNC DPM_SAFE_ARRAY DPM_VECTORCALL DPM_TARGET("fma")
#define SSE4_1_FUNC DPM_SAFE_ARRAY DPM_VECTORCALL DPM_TARGET("sse4.1")
#define SSE2_FUNC DPM_SAFE_ARRAY DPM_VECTORCALL DPM_TARGET("sse2")

namespace dpm::detail
{
	enum sincos_op
	{
		OP_SINCOS = 3,
		OP_COS = 2,
		OP_SIN = 1,
	};

	template<sincos_op Mask>
	struct sincos_ret { using type = std::pair<__m128d, __m128d>; };
	template<>
	struct sincos_ret<sincos_op::OP_SIN> { using type = __m128d; };
	template<>
	struct sincos_ret<sincos_op::OP_COS> { using type = __m128d; };
	template<sincos_op Mask>
	using sincos_ret_t = typename sincos_ret<Mask>::type;

	template<sincos_op Mask>
	inline static sincos_ret_t<Mask> DPM_FORCEINLINE return_sincos(__m128d sin, __m128d cos) noexcept
	{
		if constexpr (Mask == sincos_op::OP_SINCOS)
			return {sin, cos};
		else if constexpr (Mask == sincos_op::OP_SIN)
			return {sin};
		else
			return {cos};
	}

	inline static std::tuple<__m128d, __m128d, __m128d> DPM_FORCEINLINE prepare_sincos(__m128d x, __m128d abs_x) noexcept
	{
		/* y = |x| * 4 / Pi */
		auto y = _mm_mul_pd(abs_x, _mm_set1_pd(fopi_f64));

		/* i = isodd(y) ? y + 1 : y */
		auto i = x86_cvt_f64_i64_sse2(y);
		i = _mm_add_epi64(i, _mm_set1_epi64x(1));
		i = _mm_and_si128(i, _mm_set1_epi64x(~1ll));
		y = x86_cvt_i64_f64_sse2(i); /* y = i */

		/* Extract sign bit mask */
		const auto flip_sign = _mm_slli_epi64(_mm_and_si128(i, _mm_set1_epi64x(4)), 61);
		const auto sign = _mm_xor_pd(x86_masksign(x), std::bit_cast<__m128d>(flip_sign));

		/* Find polynomial selection mask */
		auto p_mask = std::bit_cast<__m128d>(_mm_and_si128(i, _mm_set1_epi64x(2)));
		p_mask = _mm_cmpeq_pd(p_mask, _mm_setzero_pd());

		return {y, sign, p_mask};
	}

#if defined(DPM_HAS_FMA) || defined(DPM_DYNAMIC_DISPATCH)
	template<sincos_op Mask>
	inline static sincos_ret_t<Mask> FMA_FUNC SINCOS_FUNC_SIGN(sincos_fma) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = _mm_fnmadd_pd(y, _mm_set1_pd(dp_sincos_f64[0]), x);
		z = _mm_fnmadd_pd(y, _mm_set1_pd(dp_sincos_f64[1]), z);
		z = _mm_fnmadd_pd(y, _mm_set1_pd(dp_sincos_f64[2]), z);
		const auto zz = _mm_mul_pd(z, z);

		/* p1 (0 <= a <= Pi/4) */
		auto p1 = x86_polevl_f64_fma(zz, std::span{sincof_f64});    /* p1 = sincof_f64(zz) */
		p1 = _mm_fmadd_pd(_mm_mul_pd(p1, zz), z, z);                /* p1 = p1 * zz * z + z */

		/* p2 (Pi/4 <= a <= 0) */
		auto p2 = x86_polevl_f64_fma(zz, std::span{coscof_f64});    /* p2 = coscof_f64(zz) */
		p2 = _mm_mul_pd(_mm_mul_pd(zz, p2), zz);                    /* p2 = zz * p2 * zz */
		p2 = _mm_fmadd_pd(zz, _mm_set1_pd(0.5), p2);                /* p2 = zz * 0.5 + p2 */
		p2 = _mm_sub_pd(_mm_set1_pd(1.0), p2);                      /* p2 = 1.0 - p2 */

		__m128d p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm_blendv_pd(p2, p1, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm_xor_pd(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
			p_sin = _mm_blendv_pd(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm_blendv_pd(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm_blendv_pd(p1, p2, p_mask);  /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm_xor_pd(p_cos, sign);        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_cos = nan_mask ? NaN : p_cos */
			const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
			p_cos = _mm_blendv_pd(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_cos = zero_mask ? 1.0 : p_cos */
			p_cos = _mm_blendv_pd(p_cos, _mm_set1_pd(1.0), zero_mask);
#endif
		}

		return return_sincos<Mask>(p_sin, p_cos);
	}
#endif

#if defined(DPM_HAS_SSE4_1) || defined(DPM_DYNAMIC_DISPATCH)
	template<sincos_op Mask>
	inline static sincos_ret_t<Mask> SSE4_1_FUNC SINCOS_FUNC_SIGN(sincos_sse4_1) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = x86_fnmadd_sse2(y, _mm_set1_pd(dp_sincos_f64[0]), x);
		z = x86_fnmadd_sse2(y, _mm_set1_pd(dp_sincos_f64[1]), z);
		z = x86_fnmadd_sse2(y, _mm_set1_pd(dp_sincos_f64[2]), z);
		const auto zz = _mm_mul_pd(z, z);

		/* p1 */
		auto p1 = x86_polevl_f64_sse(zz, std::span{sincof_f64});    /* p1 = sincof_f64(zz) */
		p1 = x86_fmadd_sse2(_mm_mul_pd(p1, zz), z, z);              /* p1 = p1 * zz * z + z */

		/* p2 */
		auto p2 = x86_polevl_f64_sse(zz, std::span{coscof_f64});    /* p2 = coscof_f64(zz) */
		p2 = _mm_mul_pd(_mm_mul_pd(zz, p2), zz);                    /* p2 = zz * p2 * zz */
		p2 = x86_fmadd_sse2(zz, _mm_set1_pd(0.5), p2);              /* p2 = zz * 0.5 + p2 */
		p2 = _mm_sub_pd(_mm_set1_pd(1.0), p2);                      /* p2 = 1.0 - p2 */

		__m128d p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm_blendv_pd(p1, p2, p_mask);  /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm_xor_pd(p_sin, sign);        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
			p_sin = _mm_blendv_pd(p_sin, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm_blendv_pd(p_sin, x, zero_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm_or_pd(_mm_andnot_pd(p_mask, p2), _mm_and_pd(p_mask, p1));   /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm_xor_pd(p_cos, sign);                                        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
			p_cos = _mm_blendv_pd(p_cos, nan, nan_mask);
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? 1.0 : p_sin */
			p_cos = _mm_blendv_pd(p_cos, _mm_set1_pd(1.0), zero_mask);
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
	}
#endif

	template<sincos_op Mask>
	inline static sincos_ret_t<Mask> SSE2_FUNC SINCOS_FUNC_SIGN(sincos_sse2) noexcept
	{
		const auto [y, sign, p_mask] = prepare_sincos(x, abs_x);

		auto z = _mm_sub_pd(x, _mm_mul_pd(y, _mm_set1_pd(dp_sincos_f64[0]))); /* _mm_fnmadd_pd */
		z = _mm_sub_pd(z, _mm_mul_pd(y, _mm_set1_pd(dp_sincos_f64[1])));      /* _mm_fnmadd_pd */
		z = _mm_sub_pd(z, _mm_mul_pd(y, _mm_set1_pd(dp_sincos_f64[2])));      /* _mm_fnmadd_pd */
		const auto zz = _mm_mul_pd(z, z);

		/* p1 */
		auto p1 = x86_polevl_f64_sse(zz, std::span{sincof_f64});    /* p1 = sincof_f64(zz) */
		p1 = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(p1, zz), z), z);      /* p1 = p1 * zz * z + z */

		/* p2 */
		auto p2 = x86_polevl_f64_sse(zz, std::span{coscof_f64});    /* p2 = coscof_f64(zz) */
		p2 = _mm_mul_pd(_mm_mul_pd(zz, p2), zz);                    /* p2 = zz * p2 * zz */
		p2 = _mm_add_pd(_mm_mul_pd(zz, _mm_set1_pd(0.5)), p2);      /* p2 = zz * 0.5 + p2 */
		p2 = _mm_sub_pd(_mm_set1_pd(1.0), p2);                      /* p2 = 1.0 - p2 */

		__m128d p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = _mm_or_pd(_mm_andnot_pd(p_mask, p1), _mm_and_pd(p_mask, p2));   /* p_sin = p_mask ? p2 : p1 */
			p_sin = _mm_xor_pd(p_sin, sign);                                        /* p_sin = sign ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
			p_sin = _mm_or_pd(_mm_andnot_pd(nan_mask, p_sin), _mm_and_pd(nan_mask, nan));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? x : p_sin */
			p_sin = _mm_or_pd(_mm_andnot_pd(zero_mask, p_sin), _mm_and_pd(zero_mask, x));
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = _mm_or_pd(_mm_andnot_pd(p_mask, p2), _mm_and_pd(p_mask, p1));   /* p_cos = p_mask ? p1 : p2 */
			p_cos = _mm_xor_pd(p_cos, sign);                                        /* p_cos = sign ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
			p_cos = _mm_or_pd(_mm_andnot_pd(nan_mask, p_cos), _mm_and_pd(nan_mask, nan));
#endif
#ifdef DPM_HANDLE_ERRORS
			/* p_sin = zero_mask ? 1.0 : p_sin */
			p_cos = _mm_or_pd(_mm_andnot_pd(zero_mask, p_cos), _mm_and_pd(zero_mask, _mm_set1_pd(1.0)));
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
	}

	template<sincos_op Mask>
	inline static sincos_ret_t<Mask> SSE2_FUNC DPM_FORCEINLINE impl_sincos(__m128d x) noexcept
	{
		const auto abs_x = x86_abs(x);

		/* Check for infinity, NaN & errors. */
#ifdef DPM_PROPAGATE_NAN
		const auto nan = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
		auto nan_mask = x86_isnan(x);

#ifdef DPM_HANDLE_ERRORS
		const auto inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
		const auto inf_mask = _mm_cmpeq_pd(abs_x, inf);
		nan_mask = _mm_or_pd(nan_mask, inf_mask);

		if (_mm_movemask_pd(inf_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
#endif
		if (_mm_movemask_pd(nan_mask) == 0b11) [[unlikely]]
			return return_sincos<Mask>(nan, nan);
#endif
#ifdef DPM_HANDLE_ERRORS
		const auto zero_mask = _mm_cmpeq_pd(abs_x, _mm_setzero_pd());
		if (_mm_movemask_pd(zero_mask) == 0b11) [[unlikely]]
			return return_sincos<Mask>(x, _mm_set1_pd(1.0));
#endif

#if !defined(DPM_HAS_FMA) && defined(DPM_DYNAMIC_DISPATCH)
		constinit static dispatcher sincos_disp = []()
		{
			if (cpuid::has_fma())
				return sincos_fma<Mask>;
#ifndef DPM_HAS_SSE4_1
			if (!cpuid::has_sse4_1())
				return sincos_sse2<Mask>;
#endif
			return sincos_sse4_1<Mask>;
		};
		return SINCOS_FUNC_CALL(sincos_disp);
#elif defined(DPM_HAS_FMA)
		return SINCOS_FUNC_CALL(sincos_fma<Mask>);
#elif defined(DPM_HAS_SSE4_1)
		return SINCOS_FUNC_CALL(sincos_sse4_1<Mask>);
#else
		return SINCOS_FUNC_CALL(sincos_sse2<Mask>);
#endif
	}

	std::pair<__m128d, __m128d> x86_sincos(__m128d x) noexcept
	{
		const auto [sin, cos] = impl_sincos<sincos_op::OP_SINCOS>(x);
		return {sin, cos};
	}
	__m128d x86_sin(__m128d x) noexcept { return impl_sincos<sincos_op::OP_SIN>(x); }
	__m128d x86_cos(__m128d x) noexcept { return impl_sincos<sincos_op::OP_COS>(x); }
}

#endif