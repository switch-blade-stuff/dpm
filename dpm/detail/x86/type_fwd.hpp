/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../generic/type.hpp"
#include "../utility.hpp"

#include "abi.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

namespace dpm::detail
{
	using simd_abi::detail::overload_128;
	using simd_abi::detail::overload_256;
	using simd_abi::detail::simd_abi_128;
	using simd_abi::detail::simd_abi_256;
	using simd_abi::detail::overload_any;
	using simd_abi::detail::simd_abi_any;

	template<typename>
	struct movemask_bits : std::integral_constant<std::size_t, 1> {};
	template<typename T> requires (sizeof(T) == 2)
	struct movemask_bits<T> : std::integral_constant<std::size_t, 2> {};
	template<typename T>
	inline constexpr auto movemask_bits_v = movemask_bits<T>::value;
}

#endif