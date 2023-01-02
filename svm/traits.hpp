/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "detail/define.hpp"

#ifndef SVM_USE_IMPORT

#include <concepts>
#include <utility>

#endif

namespace svm
{
	namespace simd_abi::detail
	{
		/* Customization point for architecture-specific backends. */
		template<typename>
		struct select_compatible;
		template<typename>
		struct select_native;
	}
	namespace detail
	{
		template<typename U, typename T, typename From = std::remove_cvref_t<U>>
		concept compatible_element = (std::unsigned_integral<T> && std::same_as<From, unsigned int>) || std::same_as<From, int> || std::convertible_to<From, T>;

		template<typename G, typename T, std::size_t N, std::size_t I = 0>
		static constexpr bool valid_generator() noexcept
		{
			if constexpr (I != N)
				return requires(G &&gen) {{ gen(std::integral_constant<std::size_t, I>()) } -> compatible_element<T>; } && valid_generator<G, T, N, I + 1>();
			else
				return true;
		}
		template<typename G, typename T, std::size_t N>
		concept element_generator = !compatible_element<G, T> && valid_generator<G, T, N>();
	}

	template<typename T>
	struct is_abi_tag : std::false_type {};
	template<typename T>
	inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

	template<typename T>
	struct is_simd : std::false_type {};
	template<typename T>
	inline constexpr bool is_simd_v = is_simd<T>::value;

	template<typename T>
	struct is_simd_mask : std::false_type {};
	template<typename T>
	inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

	template<typename T>
	struct is_simd_flag_type : std::false_type {};
	template<typename T>
	inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

	template<typename T, typename Abi = simd_abi::detail::select_compatible<T>::type>
	struct simd_size : std::false_type {};
	template<typename T, typename Abi = simd_abi::detail::select_compatible<T>::type>
	inline constexpr size_t simd_size_v = simd_size<T, Abi>::value;

	template<typename T, typename U = typename T::value_type>
	struct memory_alignment : std::false_type {};
	template<typename T, typename U = typename T::value_type>
	inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

	template<typename T, typename V>
	struct rebind_simd;
	template<typename T, typename V>
	using rebind_simd_t = typename rebind_simd<T, V>::type;

	template<std::size_t N, typename V>
	struct resize_simd;
	template<std::size_t N, typename V>
	using resize_simd_t = typename resize_simd<N, V>::type;
}