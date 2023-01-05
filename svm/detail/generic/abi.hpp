/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../../traits.hpp"

namespace svm
{
	namespace simd_abi
	{
		/** @brief ABI tag used to create an SIMD type containing a single (scalar) element.
		 * @example Given an instance of `simd<float, scalar>`, the resulting SIMD type will be a simple wrapper around a scalar `float`. */
		struct scalar { constexpr static int size = 1; };

		/** @brief ABI tag used to select an implementation-defined SIMD vector type guaranteeing ABI compatibility between
		 * translation units built for the current target architecture. */
		template<typename T>
		using compatible = typename detail::select_compatible<T>::type;
		/** @brief ABI tag used to select an implementation-defined SIMD vector type offering the best performance. */
		template<typename T>
		using native = typename detail::select_native<T>::type;

		SVM_DECLARE_EXT_NAMESPACE
		{
			/** @brief Extension ABI tag used to select an implementation-defined SIMD vector type with given minimum width and alignment requirements.
			 * @tparam N Amount of elements stored by the SIMD vector.
			 * @tparam Align Alignment requirement for the SIMD vector. If set to 0, uses an implementation-defined default alignment.
			 *
			 * @example Given an instance of `simd<float, aligned_size<4, 32>>`, the underlying SIMD vector type will be at least
			 * 4 `float`s wide and aligned to at least 32-byte boundary, such as `__m128` on x86 and `alignas(32) float32x4_t` on ARM NEON.
			 * @note Underlying SIMD vector type is implementation-defined and may be over-aligned and/or over-sized
			 * to meet the width and alignment requirements. */
			template<std::size_t N, std::size_t Align = 0>
			struct aligned_vector
			{
				constexpr static int alignment = Align;
				constexpr static int size = N;
			};
		}

		/** @brief ABI tag used to select an implementation-defined SIMD vector type with the given fixed width.
		 * @tparam N Amount of elements stored by the SIMD vector.
		 *
		 * @example Given an instance of `simd<float, fixed_size<4>>`, the underlying SIMD vector type will be at least
		 * 4 `float`s wide, such as `__m128` on x86 and `float32x4_t` on ARM NEON.
		 * @note Width & alignment of the underlying vector type is implementation-defined. ABI compatibility across
		 * different compilers & platforms is not guaranteed. */
		template<std::size_t N>
		using fixed_size = ext::aligned_vector<N, 0>;

		/** Implementation-defined maximum width of an SIMD vector type. Guaranteed to be at least 32. */
		template<typename T>
		inline constexpr int max_fixed_size = 32;

		/** @brief Deduces an ABI tag for given value type and size with optional hints.
		 * @tparam T Value type to deduce the ABI for.
		 * @tparam N Number of elements stored by the SIMD vector created with the deduced ABI tag.
		 * @tparam Abis Optional ABI tag hints to use for deduction. */
		template<typename T, std::size_t N, typename... Abis>
		struct deduce;
		template<typename T>
		struct deduce<T, 1> { using type = scalar; };
		template<typename T, std::size_t N>
		struct deduce<T, N> { using type = fixed_size<N>; };

		namespace detail
		{
			template<typename T, std::size_t N, typename Abi>
			concept accept_abi = simd_size_v<T, Abi> == N && std::is_default_constructible_v<simd<T, Abi>>;
		}

		template<typename T, std::size_t N, typename Abi, typename... Abis>
		struct deduce<T, N, Abi, Abis...> { using type = std::conditional_t<detail::accept_abi<T, N, Abi>, Abi, typename deduce<T, N, Abis...>::type>; };

		/** @brief Alias for `typename deduce<T, N, Abis...>::type`. */
		template<typename T, std::size_t N, typename... Abis>
		using deduce_t = typename deduce<T, N, Abis...>::type;
	}

	template<>
	struct is_abi_tag<simd_abi::scalar> : std::true_type {};
	template<std::size_t N, std::size_t Align>
	struct is_abi_tag<simd_abi::ext::aligned_vector<N, Align>> : std::true_type {};
}