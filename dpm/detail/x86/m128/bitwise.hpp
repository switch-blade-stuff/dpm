/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "utility.hpp"

namespace dpm
{
	namespace detail
	{
		template<typename V0, typename V1 = V0>
		[[nodiscard]] DPM_FORCEINLINE V0 bit_and(V0 a, V1 b) noexcept requires (sizeof(V0) == 16 && sizeof(V1) == 16)
		{
			return std::bit_cast<V0>(_mm_and_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b)));
		}
		template<typename V0, typename V1 = V0>
		[[nodiscard]] DPM_FORCEINLINE V0 bit_xor(V0 a, V1 b) noexcept requires (sizeof(V0) == 16 && sizeof(V1) == 16)
		{
			return std::bit_cast<V0>(_mm_xor_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b)));
		}
		template<typename V0, typename V1 = V0>
		[[nodiscard]] DPM_FORCEINLINE V0 bit_or(V0 a, V1 b) noexcept requires (sizeof(V0) == 16 && sizeof(V1) == 16)
		{
			return std::bit_cast<V0>(_mm_or_ps(std::bit_cast<__m128>(a), std::bit_cast<__m128>(b)));
		}
		template<typename V>
		[[nodiscard]] DPM_FORCEINLINE V bit_not(V x) noexcept requires (sizeof(V) == 16) { return bit_xor(x, setones<__m128>()); }
	}
}