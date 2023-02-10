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
	template<std::size_t N, std::size_t I = 1, typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V unwrap_polevl(V x, V y, std::span<const T, N> c) noexcept
	{
		if constexpr (I == N)
			return y;
		else
		{
			y = fmadd(y, x, fill<V>(c[I]));
			return unwrap_polevl<N, I + 1>(x, y, c);
		}
	}
	template<std::size_t N, typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V polevl(V x, std::span<const T, N> c) noexcept
	{
		return unwrap_polevl<N>(x, fill<V>(c[0]), c);
	}
}