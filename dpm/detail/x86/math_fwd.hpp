/*
 * Created by switchblade on 2023-02-01.
 */

#pragma once

#include "../generic/math.hpp"

namespace dpm::detail
{
	/* On GNU-compatible systems return 2 registers at once. */
#ifdef __GNUC__
	template<typename T0, typename T1>
	using vec2_return_t = T0;

	template<typename T0, typename T1>
	[[nodiscard]] vec2_return_t<T0, T1> DPM_FORCEINLINE vec2_return(T0 x, T1 y) noexcept requires (sizeof(T1) == 16)
	{
		/* Avoid discarding y */
		__asm__ ("movaps %0, %%xmm1" : : "x"(y));
		return x;
	}
	template<typename T0, typename T1>
	[[nodiscard]] T0 DPM_FORCEINLINE vec2_call(auto f, T0 x, T1 &out) noexcept requires (sizeof(T1) == 16)
	{
		x = f(x);
		/* Read second variable from xmm1 */
		__asm__ ("movaps %%xmm1, %0" : "=xm"(out));
		return x;
	}
#ifdef DPM_HAS_AVX
	template<typename T0, typename T1>
	[[nodiscard]] vec2_return_t<T0, T1> DPM_FORCEINLINE vec2_return(T0 x, T1 y) noexcept requires(sizeof(T1) == 32)
	{
		/* Avoid discarding y */
		__asm__ ("vmovaps %0, %%ymm1" : : "x"(y));
		return x;
	}
	template<typename T0, typename T1>
	[[nodiscard]] T0 DPM_FORCEINLINE vec2_call(auto f, T0 x, T1 &out) noexcept requires(sizeof(T1) == 32)
	{
		x = f(x);
		/* Read second variable from ymm1 */
		__asm__ ("vmovaps %%ymm1, %0" : "=xm"(out));
		return x;
	}
#endif
#else
	template<typename T0, typename T1>
	using vec2_return_t = std::pair<T0, T1>;

	template<typename T0, typename T1>
	[[nodiscard]] vec2_return_t<T0, T1> DPM_FORCEINLINE vec2_return(T0 x, T1 y) noexcept { return {x, y}; }
	template<typename T0, typename T1>
	[[nodiscard]] T0 DPM_FORCEINLINE vec2_call(auto f, T0 x, T1 &out) noexcept
	{
		const auto res = f(x);
		out = res.second;
		return res.first;
	}
#endif
}