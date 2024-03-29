/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../../traits.hpp"

namespace dpm
{
	namespace simd_abi
	{
		/** @brief ABI tag used to create an SIMD type containing a single (scalar) element.
		 * @example Given an instance of `simd<float, scalar>`, the resulting SIMD type will be a simple wrapper around a scalar `float`. */
		struct scalar
		{
			constexpr static std::size_t alignment = SIZE_MAX;
			constexpr static std::size_t size = 1;
		};

		/** @brief ABI tag used to select an implementation-defined SIMD vector type guaranteeing ABI compatibility between
		 * translation units built for the current target architecture. */
		template<typename T>
		using compatible = typename detail::select_compatible<T>::type;
		/** @brief ABI tag used to select an implementation-defined SIMD vector type offering the best performance. */
		template<typename T>
		using native = typename detail::select_native<T>::type;

		DPM_DECLARE_EXT_NAMESPACE
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
				constexpr static std::size_t alignment = Align;
				constexpr static std::size_t size = N;
			};

			/** @brief Extension ABI tag used to force use of a packed element-aligned buffer for SIMD storage.
			 * @tparam N Amount of elements stored by the buffer.
			 * @note Packed ABI will prevent the use of platform-specific optimizations and will leave any vectorization up to the compiler. */
			template<std::size_t N>
			using packed_buffer = aligned_vector<N, SIZE_MAX>;
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
		template<typename T, std::size_t N, std::size_t M, std::size_t A>
		struct deduce<T, N, ext::aligned_vector<M, A>> { using type = ext::aligned_vector<N, A>; };

		namespace detail
		{
			template<typename T, std::size_t N, typename Abi>
			concept accept_abi = simd_size_v<T, Abi> == N && std::is_default_constructible_v<simd<T, Abi>>;
		}

		template<typename T, std::size_t N, typename Abi, typename... Abis>
		struct deduce<T, N, Abi, Abis...> : deduce<T, N, Abis...> {};
		template<typename T, std::size_t N, typename Abi, typename... Abis> requires detail::accept_abi<T, N, Abi>
		struct deduce<T, N, Abi, Abis...> { using type = Abi; };

		/** @brief Alias for `typename deduce<T, N, Abis...>::type`. */
		template<typename T, std::size_t N, typename... Abis>
		using deduce_t = typename deduce<T, N, Abis...>::type;

		namespace detail
		{
			template<typename...>
			struct tag_sequence {};

			template<typename... Ts>
			struct common_impl : common_impl<tag_sequence<>, Ts...> {};

			template<std::size_t... Ns, std::size_t... As>
			struct common_impl<tag_sequence<aligned_vector<Ns, As>...>> { using type = aligned_vector<std::max({Ns...}), std::max({As...})>; };

			template<typename... Ts, typename U, typename... Us>
			struct common_impl<tag_sequence<Ts...>, U, Us...> : common_impl<tag_sequence<U, Ts...>, Us...> {};
			template<typename... Ts, typename... Us>
			struct common_impl<tag_sequence<Ts...>, scalar, Us...> : common_impl<tag_sequence<Ts...>, Us...> {};

			template<std::same_as<scalar>... Ts>
			struct common_impl<tag_sequence<>, Ts...> { using type = scalar; };
		}

		DPM_DECLARE_EXT_NAMESPACE
		{
			/** @brief Produces a common ABI tag from the specified tags.
			 *
			 * If the specified tags are the same, the member `type` is an alias for the specified abi tag.
			 * Otherwise, if the tags are of the same size but different alignment, `type` is an alias for the tag with the greatest alignment.
			 * Otherwise, `type` is an alias for the widest and most-aligned tag. */
			template<typename... Abis>
			struct common : detail::common_impl<Abis...> {};

			/** @brief Alias for `typename common<Abis...>::type`. */
			template<typename... Abis>
			using common_t = typename common<Abis...>::type;
		}
	}

	namespace detail
	{
		template<std::size_t N, std::size_t A>
		using avec = simd_abi::ext::aligned_vector<N, A>;
	}

	template<>
	struct is_abi_tag<simd_abi::scalar> : std::true_type {};
	template<std::size_t N, std::size_t Align>
	struct is_abi_tag<detail::avec<N, Align>> : std::true_type {};
}