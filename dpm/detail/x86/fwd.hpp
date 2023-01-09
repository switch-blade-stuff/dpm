/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../generic/type.hpp"
#include "../utility.hpp"

#include "abi.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm::detail
{
	using simd_abi::detail::x86_overload_all;
	using simd_abi::detail::x86_overload_sse;
	using simd_abi::detail::x86_overload_avx;
	using simd_abi::detail::x86_overload_avx512;

	/* Separate underlying implementation to allow higher-tier SIMD levels to re-use lower-tier implementations. */
	template<typename V, typename D, std::size_t N>
	struct x86_mask_impl;
	template<typename V, typename D, std::size_t N>
	struct x86_simd_impl;
}

#endif