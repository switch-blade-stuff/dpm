/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

namespace svm::detail
{
	template<typename T>
	[[nodiscard]] constexpr bool test_bit(T value, int pos) noexcept { return value & static_cast<T>(1 << pos); }
	template<typename T>
	constexpr void mask_bit(T &value, int pos, bool bit = true) noexcept { value &= ~static_cast<T>(1 << pos) | static_cast<T>(bit << pos); }
}