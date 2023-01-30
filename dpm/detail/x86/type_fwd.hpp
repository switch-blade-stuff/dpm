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
	using simd_abi::detail::x86_simd_abi_128;

#ifdef DPM_HAS_AVX
	using simd_abi::detail::x86_overload_256;
	using simd_abi::detail::x86_simd_abi_256;
#endif

	using simd_abi::detail::x86_overload_any;
	using simd_abi::detail::x86_simd_abi_any;

	template<typename, std::size_t>
	struct select_vector;
	template<>
	struct select_vector<float, 16> { using type = __m128; };

#ifdef DPM_HAS_SSE2
	template<std::integral T>
	struct select_vector<T, 16> { using type = __m128i; };
	template<>
	struct select_vector<double, 16> { using type = __m128d; };
#endif

#ifdef DPM_HAS_AVX
	template<std::integral T>
	struct select_vector<T, 32> { using type = __m256i; };
	template<>
	struct select_vector<float, 32> { using type = __m256; };
	template<>
	struct select_vector<double, 32> { using type = __m256d; };
#endif

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