/*
 * Created by switchblade on 2023-02-01.
 */

#include "trig.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

#include "except.hpp"
#include "polevl.hpp"

namespace dpm::detail
{
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

	sincos_ret<__m128> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m128 sign_x, __m128 abs_x) noexcept { return eval_sincos<float>(sign_x, abs_x); }
	sincos_ret<__m128d> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m128d sign_x, __m128d abs_x) noexcept { return eval_sincos<double>(sign_x, abs_x); }

#ifdef DPM_HAS_AVX
	sincos_ret<__m256> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m256 sign_x, __m256 abs_x) noexcept { return eval_sincos<float>(sign_x, abs_x); }
	sincos_ret<__m256d> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m256d sign_x, __m256d abs_x) noexcept { return eval_sincos<double>(sign_x, abs_x); }
#endif

	template<typename T, sincos_op Op, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE sincos_ret<V> impl_sincos(V x) noexcept
	{
		const auto sign_x = masksign<T>(x);
		auto abs_x = bit_xor(x, sign_x);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = isinf_abs(abs_x); test_mask(m))
			[[unlikely]] abs_x = except_invalid<T>(abs_x, m);
#endif
		return eval_sincos<T, Op>(sign_x, abs_x);
	}

	sincos_ret<__m128> DPM_MATHFUNC sincos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m128 DPM_MATHFUNC sin(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x).sin; }
	__m128 DPM_MATHFUNC cos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x).cos; }

	sincos_ret<__m128d> DPM_MATHFUNC sincos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SINCOS>(x); }
	__m128d DPM_MATHFUNC sin(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x).sin; }
	__m128d DPM_MATHFUNC cos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x).cos; }

#ifdef DPM_HAS_AVX
	sincos_ret<__m256> DPM_MATHFUNC sincos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SINCOS>(x); }
	__m256 DPM_MATHFUNC sin(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x).sin; }
	__m256 DPM_MATHFUNC cos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x).cos; }

	sincos_ret<__m256d> DPM_MATHFUNC sincos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SINCOS>(x); }
	__m256d DPM_MATHFUNC sin(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x).sin; }
	__m256d DPM_MATHFUNC cos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x).cos; }
#endif
}

#endif