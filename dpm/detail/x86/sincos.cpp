/*
 * Created by switchblade on 2023-02-01.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "../fconst.hpp"
#include "polevl.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
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
		const auto x_sign = masksign(x);
		const auto abs_x = bit_xor(x, x_sign);

		/* Check for infinity, NaN & errors. */
#ifdef DPM_PROPAGATE_NAN
		const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
		auto nan_mask = isunord(x, x);

#ifdef DPM_HANDLE_ERRORS
		const auto inf_mask = isinf_abs(abs_x);
		nan_mask = bit_or(nan_mask, inf_mask);
		if (movemask<T>(inf_mask)) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
		}
#endif
		if (movemask<T>(nan_mask) == fill_bits<extent>()) [[unlikely]]
			return return_sincos<Mask>(nan, nan);
#endif

		/* y = |x| * 4 / Pi */
		auto y = div<T>(abs_x, fill<V>(pio4<T>));

		/* i = isodd(y) ? y + 1 : y */
		auto i = cvtt<I, T>(y);
		i = add<I>(i, fill<decltype(i)>(I{1}));
		i = bit_and(i, fill<decltype(i)>(I{~1ll}));
		y = cvt<T, I>(i);

		/* Extract sign bit mask */
		const auto bit4 = bit_shiftl<I, sizeof(I) * 8 - 3>(bit_and(i, fill<decltype(i)>(I{4})));
		const auto bit2 = bit_shiftl<I, sizeof(I) * 8 - 2>(bit_and(i, fill<decltype(i)>(I{2})));
		[[maybe_unused]] V sign_sin = {}, sign_cos = {};
		if constexpr (Mask & OP_COS) sign_sin = bit_xor(std::bit_cast<V>(bit4), std::bit_cast<V>(bit2));
		if constexpr (Mask & OP_SIN) sign_sin = bit_xor(std::bit_cast<V>(bit4), x_sign);

		/* Polynomial selection mask !(i & 2) */
		const auto p_mask = std::bit_cast<V>(cmp_eq<I>(bit2, setzero<decltype(i)>()));

		auto z = fmadd(y, fill<V>(dp_sincos<T>[0]), abs_x);
		z = fmadd(y, fill<V>(dp_sincos<T>[1]), z);
		z = fmadd(y, fill<V>(dp_sincos<T>[2]), z);
		const auto zz = mul<T>(z, z);

		/* p1 (0 <= a <= Pi/4) */
		auto p1 = polevl(zz, std::span{sincof<T>}); /* p1 = sincof(zz) */
		p1 = fmadd(mul<T>(p1, zz), z, z);           /* p1 = p1 * zz * z + z */

		/* p2 (Pi/4 <= a <= 0) */
		auto p2 = polevl(zz, std::span{coscof<T>}); /* p2 = coscof(zz) */
		p2 = mul<T>(mul<T>(zz, p2), zz);            /* p2 = zz * p2 * zz */
		p2 = fmadd(zz, fill<V>(T{0.5}), p2);        /* p2 = zz * 0.5 + p2 */
		p2 = sub<T>(fill<V>(T{1.0}), p2);           /* p2 = 1.0 - p2 */

		V p_cos = {}, p_sin = {};
		if constexpr (Mask & sincos_op::OP_SIN)
		{
			/* Select between p1 and p2 & restore sign */
			p_sin = blendv<T>(p2, p1, p_mask);  /* p_sin = p_mask ? p1 : p2 */
			p_sin = bit_xor(p_sin, sign_sin);   /* p_sin = sign_sin ? -p_sin : p_sin */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_sin = nan_mask ? NaN : p_sin */
			p_sin = blendv<T>(p_sin, fill<V>(std::numeric_limits<T>::quiet_NaN()), nan_mask);
#endif
		}
		if constexpr (Mask & sincos_op::OP_COS)
		{
			/* Select between p1 and p2 & restore sign */
			p_cos = blendv<T>(p1, p2, p_mask);  /* p_cos = p_mask ? p2 : p1 */
			p_cos = bit_xor(p_cos, sign_cos);   /* p_cos = sign_cos ? -p_cos : p_cos */

			/* Handle errors & propagate NaN. */
#ifdef DPM_PROPAGATE_NAN
			/* p_cos = nan_mask ? NaN : p_cos */
			p_cos = blendv<T>(p_cos, fill<V>(std::numeric_limits<T>::quiet_NaN()), nan_mask);
#endif
		}
		return return_sincos<Mask>(p_sin, p_cos);
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