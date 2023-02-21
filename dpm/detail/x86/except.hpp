/*
 * Created by switchblade on 2023-02-15.
 */

#pragma once

#include "utility.hpp"
#include "addsub.hpp"
#include "muldiv.hpp"

namespace dpm::detail
{
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V except_uflow(V x, V sign, V mask) noexcept
	{
		const auto v_tiny = fill<V>(tiny<T>);
		return blendv<T>(x, mul<T>(v_tiny, bit_or(v_tiny, sign)), mask);
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V except_oflow(V x, V sign, V mask) noexcept
	{
		const auto v_huge = fill<V>(huge<T>);
		return blendv<T>(x, mul<T>(v_huge, bit_or(v_huge, sign)), mask);
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V except_invalid(V y, V x, V mask) noexcept
	{
		/* x = mask ? x - x / 0.0 : x; 0.0 / 0.0 == NaN + FE_INVALID */
		return blendv<T>(y, div<T>(sub<T>(x, x), setzero<V>()), mask);
	}
	template<typename T, int Sign = 1, typename V>
	[[nodiscard]] DPM_FORCEINLINE V except_divzero(V y, V x, V mask) noexcept
	{
		/* x = mask ? +-1.0 / x - x : x; 1.0 / 0.0 == inf + FE_DIVBYZERO */
		const auto v_one = fill<V>(Sign < 0 ? -one<T> : one<T>);
		return blendv<T>(y, div<T>(v_one, sub<T>(x, x)), mask);
	}
}