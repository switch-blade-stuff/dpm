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
	template<typename T, sincos_op Op = sincos_op::OP_SINCOS, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto eval_sincos(V sign_x, V abs_x) noexcept
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
		return return_sincos<Op>(p_sin, p_cos);
	}

	std::pair<__m128, __m128> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m128 sign_x, __m128 abs_x) noexcept { return eval_sincos<float>(sign_x, abs_x); }
	std::pair<__m128d, __m128d> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m128d sign_x, __m128d abs_x) noexcept { return eval_sincos<double>(sign_x, abs_x); }

#ifdef DPM_HAS_AVX
	std::pair<__m256, __m256> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m256 sign_x, __m256 abs_x) noexcept { return eval_sincos<float>(sign_x, abs_x); }
	std::pair<__m256d, __m256d> DPM_PRIVATE DPM_MATHFUNC eval_sincos(__m256d sign_x, __m256d abs_x) noexcept { return eval_sincos<double>(sign_x, abs_x); }
#endif

	template<typename T, sincos_op Op = sincos_op::OP_SINCOS, typename V, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE auto impl_sincos(V x) noexcept
	{
		const auto sign_x = masksign<T>(x);
		auto abs_x = bit_xor(x, sign_x);

		/* Enforce domain. */
#ifdef DPM_HANDLE_ERRORS
		if (const auto m = isinf_abs(abs_x); test_mask(m))
			[[unlikely]] abs_x = except_nan<T>(abs_x, m);
#endif
		return eval_sincos<T, Op>(sign_x, abs_x);
	}

	std::pair<__m128, __m128> sincos(__m128 x) noexcept { return impl_sincos<float>(x); }
	__m128 sin(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x); }
	__m128 cos(__m128 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x); }

	std::pair<__m128d, __m128d> sincos(__m128d x) noexcept { return impl_sincos<double>(x); }
	__m128d sin(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x); }
	__m128d cos(__m128d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x); }

#ifdef DPM_HAS_AVX
	std::pair<__m256, __m256> sincos(__m256 x) noexcept { return impl_sincos<float>(x); }
	__m256 sin(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_SIN>(x); }
	__m256 cos(__m256 x) noexcept { return impl_sincos<float, sincos_op::OP_COS>(x); }

	std::pair<__m256d, __m256d> sincos(__m256d x) noexcept { return impl_sincos<double>(x); }
	__m256d sin(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_SIN>(x); }
	__m256d cos(__m256d x) noexcept { return impl_sincos<double, sincos_op::OP_COS>(x); }
#endif
}

#endif