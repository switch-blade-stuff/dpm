/*
 * Created by switchblade on 2023-01-13.
 */

#pragma once

#include "mbase.hpp"

#ifndef DPM_USE_IMPORT

#include <span>

#endif

namespace dpm::detail
{
	template<std::size_t N, std::size_t I, std::size_t J = 0, typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V polevl(V x, V y, std::span<const T, N> c) noexcept
	{
		if constexpr (I == 0)
			return y;
		else
		{
			y = fmadd(y, x, fill<V>(c[J]));
			return polevl<N, I - 1, J + 1>(x, y, c);
		}
	}
	template<std::size_t N, typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V polevl(V x, std::span<const T, N> c) noexcept
	{
		return polevl<N, N>(x, fill<V>(c[0]), c);
	}
}