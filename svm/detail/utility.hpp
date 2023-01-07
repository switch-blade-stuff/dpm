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

	template<typename T>
	[[nodiscard]] constexpr T extend_bool(bool b) noexcept { return -static_cast<T>(b); }

	template<typename T, std::size_t N, std::size_t VSize>
	[[nodiscard]] constexpr std::size_t align_data() noexcept
	{
		const auto size_mult = VSize / sizeof(T);
		return N / size_mult + !!(N % size_mult);
	}
}