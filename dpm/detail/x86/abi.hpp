/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../generic/abi.hpp"
#include "../utility.hpp"
#include "../alias.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
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
			 * `long double` is never subject to vectorization. */
#ifdef DPM_HAS_SSE2
			template<typename T, typename U = std::decay_t<T>>
			concept has_vector = std::integral<U> || std::same_as<U, float> || std::same_as<U, double>;
#else
			template<typename T, typename U = std::decay_t<T>>
			concept has_vector = std::same_as<U, float>;
#endif

			/* Select an `aligned_vector` ABI tag that fits the specified x86 vector. */
			template<has_vector T, std::size_t Bytes>
			struct select_vector { using type = ext::aligned_vector<Bytes / sizeof(T), Bytes>; };
			template<typename T>
			using select_m128 = select_vector<T, 16>;
			template<typename T>
			using select_m256 = select_vector<T, 32>;

			template<typename, std::size_t>
			struct default_align {};

			/* Select a native x86 vector type for the specified vector size. Prefer the largest available type to enable efficient operations. */
#ifdef DPM_HAS_AVX
			template<has_vector T, std::size_t N> requires (N <= 16 / sizeof(T))
			struct default_align<T, N> : std::integral_constant<std::size_t, 16> {};
			template<has_vector T, std::size_t N>
			struct default_align<T, N> : std::integral_constant<std::size_t, 32> {};
#else
			template<has_vector T, std::size_t N>
			struct default_align<T, N> : std::integral_constant<std::size_t, 16> {};
#endif

			template<typename T, std::size_t N>
			concept has_default = 1 < N && requires { default_align<T, N>::value; };
			template<typename T, std::size_t N, std::size_t A, std::size_t VAlign, std::size_t MaxAlign>
			concept overload_simd = has_vector<T> && (A == VAlign || (MaxAlign > A && has_default<T, N> && default_align<T, N>::value == VAlign));

#ifdef DPM_HAS_AVX
			template<typename T, std::size_t N, std::size_t A = 0>
			concept x86_overload_256 = overload_simd<T, N, A, 32, SIZE_MAX>;
			template<typename T, std::size_t N, std::size_t A = 0>
			concept x86_overload_128 = overload_simd<T, N, A, 16, 32>;
#else
			template<typename T, std::size_t N, std::size_t A = 0>
			concept x86_overload_128 = overload_simd<T, N, A, 16, SIZE_MAX>;
#endif

			template<typename T, typename Abi, std::size_t A>
			struct is_simd_abi : std::false_type {};
			template<typename T, std::size_t N, std::size_t A> requires x86_overload_256<T, N, A>
			struct is_simd_abi<T, ext::aligned_vector<N, A>, 32> : std::true_type {};
			template<typename T, std::size_t N, std::size_t A> requires x86_overload_128<T, N, A>
			struct is_simd_abi<T, ext::aligned_vector<N, A>, 16> : std::true_type {};

			template<typename Abi, typename T>
			concept x86_simd_abi_128 = is_simd_abi<Abi, T, 16>::value;
			template<typename Abi, typename T>
			concept x86_simd_abi_256 = is_simd_abi<Abi, T, 32>::value;

			template<typename T, std::size_t N, std::size_t A>
			concept x86_overload_any = x86_overload_128<T, N, A> || x86_overload_256<T, N, A>;
			template<typename Abi, typename T>
			concept x86_simd_abi_any = x86_simd_abi_128<Abi, T> || x86_simd_abi_256<Abi, T>;

			/* SSE is the least common denominator for most intel CPUs since 1999 (and is a requirement for all 64-bit CPUs). */
			template<has_vector T>
			struct select_compatible<T> : select_m128<T> {};

#ifdef DPM_HAS_AVX
			template<has_vector T>
			struct select_native<T> : select_m256<T> {};
#else
			template<has_vector T>
			struct select_native<T> : select_m128<T> {};
#endif
		}

		DPM_DECLARE_EXT_NAMESPACE
		{
			/** @brief Extension ABI tag used to select SSE vectors as the underlying SIMD type. */
			template<typename T>
			using sse = typename detail::select_m128<T>::type;

#ifdef DPM_HAS_AVX
			/** @brief Extension ABI tag used to select AVX vectors as the underlying SIMD type. */
			template<typename T>
			using avx = typename detail::select_m256<T>::type;
#endif
		}

		template<typename T, std::size_t N> requires detail::x86_overload_128<T, N>
		struct deduce<T, N> { using type = ext::aligned_vector<N, 16>; };
#ifdef DPM_HAS_AVX
		template<typename T, std::size_t N> requires detail::x86_overload_256<T, N>
		struct deduce<T, N> { using type = ext::aligned_vector<N, 32>; };
#endif
	}
}

#endif