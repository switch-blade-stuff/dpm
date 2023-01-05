/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../generic/abi.hpp"

#if defined(SVM_ARCH_X86) && (defined(SVM_HAS_SSE) || defined(SVM_DYNAMIC_DISPATCH))

#include <immintrin.h>

namespace svm
{
	namespace simd_abi
	{
		namespace detail
		{
			using namespace svm::detail;

			/* `long double` is not supported by x86 SIMD. */
			template<typename T, typename U = std::decay_t<T>>
			concept has_x86_vector = std::integral<U> || std::same_as<U, float> || std::same_as<U, double>;

			/* Select an `aligned_vector` ABI tag that fits the specified x86 vector. */
			template<has_x86_vector T, typename V>
			struct select_x86_vector { using type = ext::aligned_vector<sizeof(V) / sizeof(T), alignof(V)>; };
			template<typename T>
			using select_sse = select_x86_vector<T, __m128>;
			template<typename T>
			using select_avx = select_x86_vector<T, __m256>;
			template<typename T>
			using select_avx512 = select_x86_vector<T, __m512>;

			template<typename, std::size_t>
			struct default_x86_align;

			/* Select a native x86 vector type for the specified vector size. Prefer the largest available type to enable efficient operations. */
#if (defined(SVM_HAS_AVX512) || defined(SVM_DYNAMIC_DISPATCH)) && defined(SVM_NATIVE_AVX512)
			template<has_x86_vector T, std::size_t N> requires (N <= sizeof(__m256) / sizeof(T) && N > sizeof(__m128) / sizeof(T))
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m256)> {};
			template<has_x86_vector T, std::size_t N> requires (N <= sizeof(__m128) / sizeof(T))
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m128)> {};
			template<has_x86_vector T, std::size_t N>
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m512)> {};
#elif defined(SVM_HAS_AVX) || defined(SVM_DYNAMIC_DISPATCH)
			template<has_x86_vector T, std::size_t N> requires (N <= sizeof(__m128) / sizeof(T))
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m128)> {};
			template<has_x86_vector T, std::size_t N>
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m256)> {};
#else
			template<has_x86_vector T, std::size_t N>
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m128)> {};
#endif

			template<typename T, std::size_t N>
			concept has_x86_default = has_x86_vector<T> && requires { typename default_x86_align<T, N>; };

			template<typename T, std::size_t N, std::size_t A, typename V>
			concept x86_simd_overload = vectorizable<T> && (A >= alignof(V) || A == 0) && has_x86_default<T, N> && default_x86_align<T, N>::value == alignof(V);
			template<typename T, std::size_t N, std::size_t A = 0>
			concept x86_sse_overload = x86_simd_overload<T, N, A, __m128>;
			template<typename T, std::size_t N, std::size_t A = 0>
			concept x86_avx_overload = x86_simd_overload<T, N, A, __m256>;
			template<typename T, std::size_t N, std::size_t A = 0>
			concept x86_avx512_overload = x86_simd_overload<T, N, A, __m512>;

			/* SSE is the least common denominator for most intel CPUs since 1999 (and is a requirement for all 64-bit CPUs). */
			template<has_x86_vector T>
			struct select_compatible<T> : select_sse<T> {};

#if defined(SVM_HAS_AVX512) && defined(SVM_NATIVE_AVX512)
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

		SVM_DECLARE_EXT_NAMESPACE
		{
			/** @brief Extension ABI tag used to select SSE vectors as the underlying SIMD type. */
			template<typename T>
			using sse = typename detail::select_sse<T>::type;

			/* Only enable non-SSE tags when support for them is available, or when dynamic dispatch is enabled. */
#if defined(SVM_HAS_AVX) || defined(SVM_DYNAMIC_DISPATCH)
			/** @brief Extension ABI tag used to select AVX vectors as the underlying SIMD type. */
			template<typename T>
			using avx = typename detail::select_avx<T>::type;
#endif
#if defined(SVM_HAS_AVX512) || defined(SVM_DYNAMIC_DISPATCH)
			/** @brief Extension ABI tag used to select AVX512 vectors as the underlying SIMD type. */
			template<typename T>
			using avx512 = typename detail::select_avx512<T>::type;
#endif
		}

		template<typename T, std::size_t N> requires detail::x86_sse_overload<T, N>
		struct deduce<T, N> { using type = ext::sse<T>; };
#if defined(SVM_HAS_AVX) || defined(SVM_DYNAMIC_DISPATCH)
		template<typename T, std::size_t N> requires detail::x86_avx_overload<T, N>
		struct deduce<T, N> { using type = ext::avx<T>; };
#endif
#if defined(SVM_HAS_AVX512) || defined(SVM_DYNAMIC_DISPATCH)
		template<typename T, std::size_t N> requires detail::x86_avx512_overload<T, N>
		struct deduce<T, N> { using type = ext::avx512<T>; };
#endif

		/* If AVX512 is required, use 64 bytes for max_fixed_size. Otherwise, fall back to the default 32. */
#if (defined(SVM_HAS_AVX512) || defined(SVM_DYNAMIC_DISPATCH)) && defined(SVM_NATIVE_AVX512)
		template<typename I> requires(std::integral<I> && sizeof(I) == 1)
		inline constexpr int max_fixed_size<I> = 64;
#endif
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, std::size_t N, std::size_t Align> requires simd_abi::detail::x86_sse_overload<T, N, Align>
		struct has_native_vector<T, simd_abi::ext::aligned_vector<N, Align>> : std::true_type {};
		template<typename T, std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx_overload<T, N, Align>
		struct has_native_vector<T, simd_abi::ext::aligned_vector<N, Align>> : std::true_type {};
		template<typename T, std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx512_overload<T, N, Align>
		struct has_native_vector<T, simd_abi::ext::aligned_vector<N, Align>> : std::true_type {};

		template<std::integral T, std::size_t N, std::size_t Align> requires simd_abi::detail::x86_sse_overload<T, N, Align>
		struct native_vector_type<T, simd_abi::ext::aligned_vector<N, Align>> { using type = __m128i; };
		template<std::integral T, std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx_overload<T, N, Align>
		struct native_vector_type<T, simd_abi::ext::aligned_vector<N, Align>> { using type = __m256i; };
		template<std::integral T, std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx512_overload<T, N, Align>
		struct native_vector_type<T, simd_abi::ext::aligned_vector<N, Align>> { using type = __m512i; };

		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_sse_overload<float, N, Align>
		struct native_vector_type<float, simd_abi::ext::aligned_vector<N, Align>> { using type = __m128; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx_overload<float, N, Align>
		struct native_vector_type<float, simd_abi::ext::aligned_vector<N, Align>> { using type = __m256; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx512_overload<float, N, Align>
		struct native_vector_type<float, simd_abi::ext::aligned_vector<N, Align>> { using type = __m512; };

		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_sse_overload<double, N, Align>
		struct native_vector_type<double, simd_abi::ext::aligned_vector<N, Align>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx_overload<double, N, Align>
		struct native_vector_type<double, simd_abi::ext::aligned_vector<N, Align>> { using type = __m256d; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_avx512_overload<double, N, Align>
		struct native_vector_type<double, simd_abi::ext::aligned_vector<N, Align>> { using type = __m512d; };
	}
}

#endif