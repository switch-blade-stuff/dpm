/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../generic/abi.hpp"
#include "../utility.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include <immintrin.h>

namespace dpm
{
	namespace simd_abi
	{
		namespace detail
		{
			using namespace dpm::detail;

			/* Non-f32 types are only vectorizable if SSE2 is available. This is so even with dynamic dispatch,
			 * since dispatched functions have a minimum required SIMD level due to ABI and inlining issues.
			 * I.E. you the compiler cannot generate non-SSE2 code for SSE2 data types.
			 *
			 * `long double` is never subject to vectorization. */
#ifdef DPM_HAS_SSE2
			template<typename T, typename U = std::decay_t<T>>
			concept has_x86_vector = std::integral<U> || std::same_as<U, float> || std::same_as<U, double>;
#else
			template<typename T, typename U = std::decay_t<T>>
			concept has_x86_vector = std::same_as<U, float>;
#endif

			/* Select an `aligned_vector` ABI tag that fits the specified x86 vector. */
			template<has_x86_vector T, std::size_t Size, std::size_t Align>
			struct select_x86_vector { using type = ext::aligned_vector<Size / sizeof(T), Align>; };
			template<typename T>
			using select_m128 = select_x86_vector<T, sizeof(__m128), alignof(__m128)>;
			template<typename T>
			using select_m256 = select_x86_vector<T, sizeof(__m256), alignof(__m256)>;
			template<typename T>
			using select_m512 = select_x86_vector<T, sizeof(__m512), alignof(__m512)>;

			template<typename, std::size_t>
			struct default_x86_align;

			/* Select a native x86 vector type for the specified vector size. Prefer the largest available type to enable efficient operations. */
#if (defined(DPM_HAS_AVX512) || defined(DPM_DYNAMIC_DISPATCH)) && defined(DPM_NATIVE_AVX512)
			template<has_x86_vector T, std::size_t N> requires (N <= sizeof(__m256) / sizeof(T) && N > sizeof(__m128) / sizeof(T))
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m256)> {};
			template<has_x86_vector T, std::size_t N> requires (N <= sizeof(__m128) / sizeof(T))
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m128)> {};
			template<has_x86_vector T, std::size_t N>
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m512)> {};
#elif defined(DPM_HAS_AVX) || defined(DPM_DYNAMIC_DISPATCH)
			template<has_x86_vector T, std::size_t N> requires (N <= sizeof(__m128) / sizeof(T))
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m128)> {};
			template<has_x86_vector T, std::size_t N>
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m256)> {};
#else
			template<has_x86_vector T, std::size_t N>
			struct default_x86_align<T, N> : std::integral_constant<std::size_t, alignof(__m128)> {};
#endif

			template<typename T, std::size_t N>
			concept has_x86_default = has_x86_vector<T> && N > 1 && requires
		{
			typename default_x86_align<T, N>;
		};
		template<typename T, std::size_t N, std::size_t A, std::size_t VAlign, std::size_t MaxAlign>
		concept x86_overload_simd = vectorizable<T> && (A == VAlign || (MaxAlign > A && has_x86_default<T, N> && default_x86_align<T, N>::value == VAlign));

		template<typename T, std::size_t N, std::size_t A = 0>
		concept x86_overload_m512 = x86_overload_simd<T, N, A, alignof(__m512), std::numeric_limits<std::size_t>::max()>;
#ifdef DPM_NATIVE_AVX512
		template<typename T, std::size_t N, std::size_t A = 0>
		concept x86_overload_m256 = x86_overload_simd<T, N, A, alignof(__m256), alignof(__m512)>;
#else
		template<typename T, std::size_t N, std::size_t A = 0>
		concept x86_overload_m256 = x86_overload_simd<T, N, A, alignof(__m256), std::numeric_limits<std::size_t>::max()>;
#endif
		template<typename T, std::size_t N, std::size_t A = 0>
		concept x86_overload_m128 = x86_overload_simd<T, N, A, alignof(__m128), alignof(__m256)>;

		template<typename T, std::size_t N, std::size_t A = 0>
		concept x86_overload_all = x86_overload_m128<T, N, A> || x86_overload_m256<T, N, A> || x86_overload_m512<T, N, A>;

		/* SSE is the least common denominator for most intel CPUs since 1999 (and is a requirement for all 64-bit CPUs). */
		template<has_x86_vector T>
		struct select_compatible<T> : select_m128<T> {};

#if defined(DPM_HAS_AVX512) && defined(DPM_NATIVE_AVX512)
		template<has_x86_vector T>
		struct select_native<T> : select_m512<T> {};
#elif defined(DPM_HAS_AVX)
		template<has_x86_vector T>
		struct select_native<T> : select_m256<T> {};
#else
		template<has_x86_vector T>
		struct select_native<T> : select_m128<T> {};
#endif
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** @brief Extension ABI tag used to select SSE vectors as the underlying SIMD type. */
		template<typename T>
		using sse = typename detail::select_m128<T>::type;

		/* Only enable non-SSE tags when support for them is available, or when dynamic dispatch is enabled. */
#if defined(DPM_HAS_AVX) || defined(DPM_DYNAMIC_DISPATCH)
		/** @brief Extension ABI tag used to select AVX vectors as the underlying SIMD type. */
		template<typename T>
		using avx = typename detail::select_m256<T>::type;
#endif
#if defined(DPM_HAS_AVX512) || defined(DPM_DYNAMIC_DISPATCH)
		/** @brief Extension ABI tag used to select AVX512 vectors as the underlying SIMD type. */
		template<typename T>
		using avx512 = typename detail::select_m512<T>::type;
#endif
	}

	template<typename T, std::size_t N> requires detail::x86_overload_m128<T, N>
	struct deduce<T, N> { using type = ext::aligned_vector<N, alignof(__m128)>; };
#if defined(DPM_HAS_AVX) || defined(DPM_DYNAMIC_DISPATCH)
	template<typename T, std::size_t N> requires detail::x86_overload_m256<T, N>
	struct deduce<T, N> { using type = ext::aligned_vector<N, alignof(__m256)>; };
#endif
#if defined(DPM_HAS_AVX512) || defined(DPM_DYNAMIC_DISPATCH)
	template<typename T, std::size_t N> requires detail::x86_overload_m512<T, N>
	struct deduce<T, N> { using type = ext::aligned_vector<N, alignof(__m512)>; };
#endif

	/* If AVX512 is required, use 64 bytes for max_fixed_size. Otherwise, fall back to the default 32. */
#if (defined(DPM_HAS_AVX512) || defined(DPM_DYNAMIC_DISPATCH)) && defined(DPM_NATIVE_AVX512)
	template<typename I> requires(std::integral<I> && sizeof(I) == 1)
	inline constexpr int max_fixed_size<I> = 64;
#endif
}
}

#endif