/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../generic/abi.hpp"

#if defined(DPM_ARCH_ARM) && defined(DPM_HAS_NEON)

#include <arm_neon.h>

namespace dpm
{
	namespace simd_abi
	{
		namespace detail
		{
			/* `long double` is not supported by x86 SIMD. */
			template<typename T, typename U = std::decay_t<T>>
			concept has_neon_vector = std::integral<U> || std::same_as<U, float> || std::same_as<U, double>;

			/* Select an `aligned_vector` ABI tag that fits the specified NEON vector. */
			template<has_neon_vector T>
			struct select_neon;

			template<typename I> requires (std::signed_integral<I> && sizeof(T) == 1)
			struct select_neon { using type = ext::aligned_vector<16, alignof(int8x16_t)>; };
			template<typename I> requires (std::signed_integral<I> && sizeof(T) == 2)
			struct select_neon { using type = ext::aligned_vector<8, alignof(int16x8_t)>; };
			template<typename I> requires (std::signed_integral<I> && sizeof(T) == 4)
			struct select_neon { using type = ext::aligned_vector<4, alignof(int32x4_t)>; };

			template<typename I> requires (std::unsigned_integral<I> && sizeof(T) == 1)
			struct select_neon { using type = ext::aligned_vector<16, alignof(uint8x16_t)>; };
			template<typename I> requires (std::unsigned_integral<I> && sizeof(T) == 2)
			struct select_neon { using type = ext::aligned_vector<8, alignof(uint16x8_t)>; };
			template<typename I> requires (std::unsigned_integral<I> && sizeof(T) == 4)
			struct select_neon { using type = ext::aligned_vector<4, alignof(uint32x4_t)>; };

			template<>
			struct select_neon<float> { using type = ext::aligned_vector<4, alignof(float32x4_t)>; };

#ifdef DPM_ARCH_ARM64
			template<typename I> requires(std::signed_integral<I> && sizeof(T) == 8)
			struct select_neon { using type = ext::aligned_vector<2, alignof(int64x2_t)>; };
			template<typename I> requires(std::unsigned_integral<I> && sizeof(T) == 8)
			struct select_neon { using type = ext::aligned_vector<2, alignof(uint64x2_t)>; };

			template<>
			struct select_neon<double> { using type = ext::aligned_vector<2, alignof(float64x2_t)>; };
#endif

			template<has_neon_vector T>
			struct select_compatible<T> : select_neon<T> {};
			template<has_neon_vector T>
			struct select_native<T> : select_neon<T> {};
		}

		DPM_DECLARE_EXT_NAMESPACE
		{
			/** @brief Extension ABI tag used to select ARM NEON vectors as the underlying SIMD type. */
			template<typename T>
			using neon = typename detail::select_neon<T>::type;
		}
	}
}

/* Undefine the NEON macros if we forced NEON support. */
#if !defined(DPM_HAS_NEON) && (defined(__clang__) || defined(__GNUC__))
#undef __ARM_NEON
#undef __ARM_NEON__
#endif

#endif