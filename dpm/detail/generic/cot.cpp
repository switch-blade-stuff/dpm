/*
 * Created by switchblade on 2023-02-05.
 */

#include "../fconst.hpp"
#include "math.hpp"

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

namespace dpm::detail
{
	template<typename>
	struct cot_int;
	template<>
	struct cot_int<long double> { using type = std::int32_t; };
	template<>
	struct cot_int<double> { using type = std::int64_t; };
	template<>
	struct cot_int<float> { using type = std::int32_t; };

	template<typename T, typename I = typename cot_int<T>::type>
	[[nodiscard]] DPM_FORCEINLINE T impl_cot(T x) noexcept
	{
		T abs_x, x_sign;
		if constexpr (!std::same_as<T, long double>)
		{
			const auto i_sign = std::bit_cast<I>(x) & std::bit_cast<I>(T{-0.0});
			abs_x = std::bit_cast<T>(std::bit_cast<I>(x) ^ i_sign);
			x_sign = std::bit_cast<T>(i_sign);
		}
		else
		{
			abs_x = std::fabs(x);
			x_sign = x;
		}
		if (abs_x == T{0.0}) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
			return std::numeric_limits<T>::quiet_NaN();
		}

		/* y = |x| * 4 / Pi */
		auto y = abs_x * fopi<T>;

		/* j = isodd(y) ? y + 1 : y */
		auto j = I{};
		if constexpr (std::same_as<T, long double>)
		{
			const auto z = y - std::floor(y / 16) * 16;
			if ((j = static_cast<std::int32_t>(z)) & 1)
			{
				y += 1.0L;
				j += 1;
			}
		}
		else
		{
			j = (static_cast<I>(std::trunc(y)) + 1ll) & ~1ll;
			y = static_cast<T>(j);
		}

		/* Calculate & adjust cotangent polynomial. */
		y = ((abs_x + y * dp_tancot<T>[0]) + y * dp_tancot<T>[1]) + y * dp_tancot<T>[2];
		if (const auto y2 = y * y; y2 > tancot_pmin<T>)
		{
			auto p0 = y2 * ((y2 * tancot_p<T>[0] + tancot_p<T>[1]) * y2 + tancot_p<T>[2]);
			auto p1 = ((y2 * tancot_q<T>[0] + tancot_q<T>[1]) * y2 + tancot_q<T>[2]);
			p1 = (p1 * y2 + tancot_q<T>[3]) * y2 + tancot_q<T>[4];
			y += y * (p0 / p1);
		}
		y = (j & 2) ? -y : T{1.0} / y;

		/* Restore sign. */
		if constexpr (!std::same_as<T, long double>)
			return std::bit_cast<T>(std::bit_cast<I>(y) ^ std::bit_cast<I>(x_sign));
		else
			return std::copysign(y, x_sign);
	}

	[[nodiscard]] float DPM_PUBLIC DPM_MATHFUNC cot(float x) noexcept { return impl_cot(x); }
	[[nodiscard]] double DPM_PUBLIC DPM_MATHFUNC cot(double x) noexcept { return impl_cot(x); }
	[[nodiscard]] long double DPM_PUBLIC DPM_MATHFUNC cot(long double x) noexcept { return impl_cot(x); }
}