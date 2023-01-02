/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include <arm_neon.h>
#ifdef SVM_ARCH_ARM


namespace svm::detail::abi
{
	template<typename T>
	struct neon;

	template<typename T>
	using compatible = typename detail::valid_or<neon<T>, scalar>::type;
	template<typename T>
	using native = compatible<T>;

	template<typename T> requires(std::signed_integral<T> && sizeof(T) == 1)
	struct neon<T> { using type = int8x16_t; };
	template<typename T> requires(std::signed_integral<T> && sizeof(T) == 2)
	struct neon<T> { using type = int16x8_t; };
	template<typename T> requires(std::signed_integral<T> && sizeof(T) == 4)
	struct neon<T> { using type = int32x4_t; };
	template<typename T> requires(std::signed_integral<T> && sizeof(T) == 8)
	struct neon<T> { using type = int64x2_t; };

	template<typename T> requires(std::unsigned_integral<T> && sizeof(T) == 1)
	struct neon<T> { using type = uint8x16_t; };
	template<typename T> requires(std::unsigned_integral<T> && sizeof(T) == 2)
	struct neon<T> { using type = uint16x8_t; };
	template<typename T> requires(std::unsigned_integral<T> && sizeof(T) == 4)
	struct neon<T> { using type = uint32x4_t; };
	template<typename T> requires(std::unsigned_integral<T> && sizeof(T) == 8)
	struct neon<T> { using type = uint64x2_t; };

	template<>
	struct neon<float> { using type = float32x4_t; };
	template<>
	struct neon<double> { using type = float64x2_t; };

	template<typename T>
	inline constexpr int max_fixed_size = 32;
}

#endif