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
	[[nodiscard]] DPM_FORCEINLINE double cot_poly(double abs_x, double y) noexcept
	{
		y = ((abs_x + y * dp_tancotd[0]) + y * dp_tancotd[1]) + y * dp_tancotd[2];
		if (const auto y2 = y * y; y2 > 1.0e-14)
		{
			auto p0 = y2 * ((y2 * tancot_pd[0] + tancot_pd[1]) * y2 + tancot_pd[2]);
			auto p1 = ((y2 * tancot_qd[0] + tancot_qd[1]) * y2 + tancot_qd[2]);
			p1 = (p1 * y2 + tancot_qd[3]) * y2 + tancot_qd[4];
			y += y * (p0 / p1);
		}
		return y;
	}
	[[nodiscard]] DPM_FORCEINLINE float cot_poly(float abs_x, float y) noexcept
	{
		y = ((abs_x + y * dp_tancotf[0]) + y * dp_tancotf[1]) + y * dp_tancotf[2];
		if (const auto y2 = y * y; y2 > 1.0e-4f)
		{
			const auto p = (((tancot_pf[0] * y2 + tancot_pf[1]) * y2 + tancot_pf[2]) * y2 + tancot_pf[3]);
			y += ((p * y2 + tancot_pf[4]) * y2 + tancot_pf[5]) * y2 * y;
		}
		return y;
	}

	template<typename T, typename I = int_of_size_t<sizeof(T)>>
	[[nodiscard]] DPM_FORCEINLINE T impl_cot(T x) noexcept
	{
		const auto x_sign = std::bit_cast<I>(x) & std::bit_cast<I>(T{-0.0});
		const auto abs_x = std::bit_cast<T>(std::bit_cast<I>(x) ^ x_sign);
		if (abs_x == T{0.0}) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
			return std::numeric_limits<T>::quiet_NaN();
		}

		/* y = |x| * 4 / Pi */
		auto y = abs_x * fopi<T>;

		/* j = isodd(y) ? y + 1 : y */
		const auto j = (static_cast<I>(std::trunc(y)) + 1ll) & ~1ll;
		y = static_cast<T>(j);

		/* Calculate cotangent polynomial. */
		y = cot_poly(abs_x, y);
		y = (j & 2) ? -y : T{1.0} / y;

		/* Restore sign. */
		return std::bit_cast<T>(std::bit_cast<I>(y) ^ x_sign);
	}

	[[nodiscard]] float DPM_PUBLIC DPM_MATHFUNC cot(float x) noexcept { return impl_cot(x); }
	[[nodiscard]] double DPM_PUBLIC DPM_MATHFUNC cot(double x) noexcept { return impl_cot(x); }

	[[nodiscard]] long double DPM_PUBLIC DPM_MATHFUNC cot(long double x) noexcept
	{
		const auto abs_x = std::abs(x);
		if (abs_x == 0.0L) [[unlikely]]
		{
			std::feraiseexcept(FE_INVALID);
			errno = EDOM;
			return std::numeric_limits<long double>::quiet_NaN();
		}

		/* y = |x| * 4 / Pi */
		auto y = abs_x * fopi<long double>;
		/* Extract 15 bits of integer part. */
		const auto z = y - std::floor(y / 16) * 16;     /* y - 16 * (y / 16) */
		auto j = static_cast<std::int32_t>(z);
		if (j & 1)
		{
			y += 1.0L;
			j += 1;
		}

		/* Calculate cotangent polynomial. */
		y = ((abs_x + y * dp_tancotld[0]) + y * dp_tancotld[1]) + y * dp_tancotld[2];
		if (const auto y2 = y * y; y2 > 1.0e-20L)
		{
			auto p0 = y2 * ((y2 * tancot_pld[0] + tancot_pld[1]) * y2 + tancot_pld[2]);
			auto p1 = ((y2 * tancot_qld[0] + tancot_qld[1]) * y2 + tancot_qld[2]);
			p1 = (p1 * y2 + tancot_qld[3]) * y2 + tancot_qld[4];
			y += y * (p0 / p1);
		}
		y = (j & 2) ? -y : 1.0L / y;

		/* Restore sign. */
		return std::copysign(y, x);
	}
}