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

	template<typename T, std::size_t>
	struct select_vector;
}

#endif