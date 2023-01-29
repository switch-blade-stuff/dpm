/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../generic/type.hpp"
#include "../utility.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "abi.hpp"

namespace dpm::detail
{
	using simd_abi::detail::x86_overload_128;
	using simd_abi::detail::x86_overload_256;
	using simd_abi::detail::x86_simd_abi_128;
	using simd_abi::detail::x86_simd_abi_256;
	using simd_abi::detail::x86_overload_any;
	using simd_abi::detail::x86_simd_abi_any;

	template<typename>
	struct movemask_bits : std::integral_constant<std::size_t, 1> {};
	template<typename T> requires (sizeof(T) == 2)
	struct movemask_bits<T> : std::integral_constant<std::size_t, 2> {};
	template<typename T>
	inline constexpr auto movemask_bits_v = movemask_bits<T>::value;

	template<typename T, std::size_t N, std::size_t A>
	using x86_mask = simd_mask<T, avec<N, A>>;
	template<typename T, std::size_t N, std::size_t A>
	using x86_simd = simd<T, avec<N, A>>;
}

#endif