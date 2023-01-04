/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

namespace svm::detail
{
	template<typename T>
	[[nodiscard]] constexpr bool test_bit(T value, int pos) noexcept { return value & static_cast<T>(1 << pos); }
	template<typename T>
	constexpr void mask_bit(T &value, int pos, bool bit = true) noexcept { value &= ~static_cast<T>(!bit << pos); }

	template<typename T, std::size_t N, typename V>
	[[nodiscard]] constexpr std::size_t vector_array_size() noexcept
	{
		const auto size_mult = sizeof(V) / sizeof(T);
		return N / size_mult + !!(N % size_mult);
	}
}