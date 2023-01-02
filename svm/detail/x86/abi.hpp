/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../generic/abi.hpp"

#ifdef SVM_ARCH_X86

#include "../utility.hpp"

#ifndef SVM_USE_IMPORT

#include <concepts>

#endif

#include <immintrin.h>

namespace svm::simd_abi
{
	namespace detail
	{
		/* `long double` is not supported by x86 SIMD. */
		template<typename T, typename U = std::decay_t<T>>
		concept has_x86_vector = std::integral<U> || std::same_as<U, float> || std::same_as<U, double>;

		/* Select an `aligned_vector` ABI tag that fits the specified x86 vector. */
		template<has_x86_vector T, typename V>
		struct select_x86_vector { using type = aligned_vector<sizeof(V) / sizeof(T), alignof(V)>; };
		template<typename T>
		using select_sse = select_x86_vector<T, __m128>;
		template<typename T>
		using select_avx = select_x86_vector<T, __m256>;
		template<typename T>
		using select_avx512 = select_x86_vector<T, __m512>;

		/* SSE is the least common denominator for most intel CPUs since 1999. */
		template<has_x86_vector T>
		struct select_compatible<T> : select_sse<T> {};

#if defined(SVM_NATIVE_AVX512) && defined(SVM_HAS_AVX512)
		template<has_x86_vector T>
		struct select_native<T> : select_avx512<T> {};
#elif defined(SVM_HAS_AVX)
		template<has_x86_vector T>
		struct select_native<T> : select_avx<T> {};
#else
		template<has_x86_vector T>
		struct select_native<T> : select_sse<T> {};
#endif
	}

	/** @brief Extension ABI tag used to select SSE vectors as the underlying SIMD type. */
	template<typename T>
	using sse = typename detail::select_sse<T>::type;
	/** @brief Extension ABI tag used to select AVX vectors as the underlying SIMD type. */
	template<typename T>
	using avx = typename detail::select_avx<T>::type;
	/** @brief Extension ABI tag used to select AVX512 vectors as the underlying SIMD type. */
	template<typename T>
	using avx512 = typename detail::select_avx512<T>::type;

#if defined(SVM_NATIVE_AVX512) && defined(SVM_HAS_AVX512) /* If AVX512 is required, use 64 bytes for max_fixed_size. Otherwise, fall back to the default 32. */
	template<typename I> requires(std::integral<I> && sizeof(I) == 1)
	inline constexpr int max_fixed_size<I> = 64;
#endif
}

#endif