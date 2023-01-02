/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "define.hpp"

#ifndef SVM_USE_IMPORT

#include <type_traits>

#endif

namespace svm::detail
{
	template<typename T, typename = void>
	struct is_valid : std::false_type {};
	template<typename T>
	struct is_valid<T, std::void_t<T>> : std::true_type { using type = T; };

	template<typename T, typename... Ts>
	struct valid_or
	{
		using type = std::conditional_t<is_valid<T>::value, T, typename valid_or<Ts...>::type>;
	};
	template<typename T>
	struct valid_or<T> : is_valid<T> {};

	template<typename... Ts>
	struct select_abi
	{
	};

	template<typename T>
	[[nodiscard]] constexpr bool test_bit(T value, int pos) noexcept { return value & static_cast<T>(1 << pos); }
}