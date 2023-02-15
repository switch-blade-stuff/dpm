/*
 * Created by switchblade on 2023-02-15.
 */

#pragma once

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

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
	[[nodiscard]] DPM_FORCEINLINE V except_nan(V x, V mask) noexcept
	{
		/* x = mask ? x - x / 0.0 : x; 0.0 / 0.0 == NaN + FE_INVALID */
		return blendv<T>(x, div<T>(sub<T>(x, x), setzero<V>()), mask);
	}
	template<typename T, int Sign = 1, typename V>
	[[nodiscard]] DPM_FORCEINLINE V except_inf(V x, V mask) noexcept
	{
		const auto v_one = fill<V>(Sign < 0 ? -one<T> : one<T>);
		return blendv<T>(x, div<T>(v_one, sub<T>(x, x)), mask);
	}
}