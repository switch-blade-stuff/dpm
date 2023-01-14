/*
 * Created by switchblade on 2023-01-14.
 */

#pragma once

#include "type.hpp"

#ifndef DPM_USE_IMPORT

#include <cmath>

#endif

namespace dpm
{
	template<typename T, typename Abi>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<T, Abi> sin(const simd<T, Abi> &x) noexcept
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::sin(x[i]);
		return result;
	}
	template<typename T, typename Abi>
	[[nodiscard]] inline DPM_SAFE_ARRAY simd<T, Abi> cos(const simd<T, Abi> &x) noexcept
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::cos(x[i]);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, typename Abi>
		inline DPM_SAFE_ARRAY void sincos(const simd<T, Abi> &x, simd<T, Abi> &out_sin, simd<T, Abi> &out_cos) noexcept
		{
			out_sin = sin(x);
			out_cos = cos(x);
		}
	}
}