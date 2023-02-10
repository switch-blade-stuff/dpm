/*
 * Created by switchblade on 2023-02-01.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "polevl.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
	template<typename T, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE std::pair<V, V> eval_sincos(V sign_x, V abs_x) noexcept
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

		V p_sin, p_cos;
		/* Select between p1 and p2 & restore sign */
		p_sin = blendv<T>(p1, p2, p_mask);  /* p_sin = p_mask ? p2 : p1 */
		p_sin = bit_xor(p_sin, sign_sin);   /* p_sin = sign_sin ? -p_sin : p_sin */
		/* Select between p1 and p2 & restore sign */
		p_cos = blendv<T>(p2, p1, p_mask);  /* p_cos = p_mask ? p1 : p2 */
		p_cos = bit_xor(p_cos, sign_cos);   /* p_cos = sign_cos ? -p_cos : p_cos */
		return {p_sin, p_cos};
	}

	std::pair<__m128, __m128> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m128 sign_x, __m128 abs_x) noexcept { return eval_sincos<float>(sign_x, abs_x); }
	std::pair<__m128d, __m128d> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m128d sign_x, __m128d abs_x) noexcept { return eval_sincos<double>(sign_x, abs_x); }

#ifdef DPM_HAS_AVX
	std::pair<__m256, __m256> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m256 sign_x, __m256 abs_x) noexcept { return eval_sincos<float>(sign_x, abs_x); }
	std::pair<__m256d, __m256d> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m256d sign_x, __m256d abs_x) noexcept { return eval_sincos<double>(sign_x, abs_x); }
#endif

	enum sincos_op
	{
		OP_SINCOS = 3,
		OP_COS = 2,
		OP_SIN = 1,
	};

	template<typename T, sincos_op Mask>
	struct sincos_ret { using type = T; };
	template<typename T>
	struct sincos_ret<T, sincos_op::OP_SINCOS> { using type = std::pair<T, T>; };
	template<typename T, sincos_op Mask>
	using sincos_ret_t = typename sincos_ret<T, Mask>::type;

	template<sincos_op Mask, typename V>
	[[nodiscard]] DPM_FORCEINLINE sincos_ret_t<V, Mask> return_sincos(V sin, V cos) noexcept
	{
		if constexpr (Mask == sincos_op::OP_SINCOS)
			return {sin, cos};
		else if constexpr (Mask == sincos_op::OP_SIN)
			return sin;
		else
			return cos;
	}
	template<typename T, sincos_op Mask, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_sincos(V x) noexcept
	{
		constexpr auto extent = sizeof(V) / sizeof(T);
		const auto sign_x = masksign<T>(x);
		auto abs_x = bit_xor(x, sign_x);

		/* Check for domain & NaN. */
#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
#ifdef DPM_HANDLE_ERRORS
		const auto dom_mask = isinf_abs(abs_x);
		if (movemask<T>(dom_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
		const auto nan_mask = bit_or(dom_mask, isunord(x, x));
#else
		const auto nan_mask = isunord(x, x);
#endif
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		if (movemask<T>(nan_mask) == fill_bits<extent>()) [[unlikely]]
			return return_sincos<Mask>(nan, nan);
		abs_x = bit_andnot(nan_mask, abs_x);
#endif

		/* Evaluate both sin(x) & cos(x). */
		[[maybe_unused]] auto [sin_x, cos_x] = eval_sincos<T>(sign_x, abs_x);
		/* Handle errors & propagate NaN. */
#if defined(DPM_HANDLE_ERRORS) || defined(DPM_PROPAGATE_NAN)
		if constexpr (Mask & sincos_op::OP_SIN) sin_x = blendv<T>(sin_x, nan, nan_mask);
		if constexpr (Mask & sincos_op::OP_COS) cos_x = blendv<T>(cos_x, nan, nan_mask);
#endif
		return return_sincos<Mask>(sin_x, cos_x);
	}

	std::pair<__m128, __m128> DPM_PUBLIC DPM_MATHFUNC sincos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m128 DPM_PUBLIC DPM_MATHFUNC sin(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x); }
	__m128 DPM_PUBLIC DPM_MATHFUNC cos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x); }

	std::pair<__m128d, __m128d> DPM_PUBLIC DPM_MATHFUNC sincos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SINCOS>(x); }
	__m128d DPM_PUBLIC DPM_MATHFUNC sin(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x); }
	__m128d DPM_PUBLIC DPM_MATHFUNC cos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x); }

#ifdef DPM_HAS_AVX
	std::pair<__m256, __m256> DPM_PUBLIC DPM_MATHFUNC sincos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m256 DPM_PUBLIC DPM_MATHFUNC sin(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x); }
	__m256 DPM_PUBLIC DPM_MATHFUNC cos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x); }

	std::pair<__m256d, __m256d> DPM_PUBLIC DPM_MATHFUNC sincos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SINCOS>(x); }
	__m256d DPM_PUBLIC DPM_MATHFUNC sin(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x); }
	__m256d DPM_PUBLIC DPM_MATHFUNC cos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x); }
#endif
}

#endif