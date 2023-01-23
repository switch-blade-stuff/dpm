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
	using simd_abi::detail::x86_overload_m128;
	using simd_abi::detail::x86_overload_m256;
	using simd_abi::detail::x86_simd_abi_m128;
	using simd_abi::detail::x86_simd_abi_m256;
	using simd_abi::detail::x86_overload_any;
	using simd_abi::detail::x86_simd_abi_any;
}

#endif