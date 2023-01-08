/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../generic/type.hpp"
#include "../utility.hpp"

#include "abi.hpp"

#if defined(SVM_ARCH_X86) && (defined(SVM_HAS_SSE) || defined(SVM_DYNAMIC_DISPATCH))

namespace svm::detail
{
	using simd_abi::detail::x86_overload_all;
	using simd_abi::detail::x86_overload_sse;
	using simd_abi::detail::x86_overload_avx;
	using simd_abi::detail::x86_overload_avx512;

	/* Separate underlying implementation to allow higher-tier SIMD levels to re-use lower-tier implementations. */
	template<typename V, typename D, std::size_t N>
	struct x86_impl;
}

#endif