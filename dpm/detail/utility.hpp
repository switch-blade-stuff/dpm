/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "define.hpp"

namespace dpm::detail
{
	template<std::size_t N>
	struct int_of_size;

	template<>
	struct int_of_size<8> { using type = std::int8_t; };
	template<>
	struct int_of_size<16> { using type = std::int16_t; };
	template<>
	struct int_of_size<32> { using type = std::int32_t; };
	template<>
	struct int_of_size<64> { using type = std::int64_t; };

	template<std::size_t N>
	using int_of_size_t = typename int_of_size<N>::type;

	template<std::size_t N>
	struct uint_of_size;

	template<>
	struct uint_of_size<8> { using type = std::uint8_t; };
	template<>
	struct uint_of_size<16> { using type = std::uint16_t; };
	template<>
	struct uint_of_size<32> { using type = std::uint32_t; };
	template<>
	struct uint_of_size<64> { using type = std::uint64_t; };

	template<std::size_t N>
	using uint_of_size_t = typename uint_of_size<N>::type;

	template<typename I, std::size_t N>
	concept integral_of_size = std::integral<I> && sizeof(I) == N;
	template<typename I, std::size_t N>
	concept signed_integral_of_size = std::signed_integral<I> && sizeof(I) == N;
	template<typename I, std::size_t N>
	concept unsigned_integral_of_size = std::unsigned_integral<I> && sizeof(I) == N;

	template<typename T>
	[[nodiscard]] constexpr bool test_bit(T x, int pos) noexcept { return x & static_cast<T>(1 << pos); }
	template<typename T>
	constexpr void mask_bit(T &x, int pos, bool bit = true) noexcept { x &= ~static_cast<T>(!bit << pos); }

	template<typename T>
	[[nodiscard]] constexpr T extend_bool(bool b) noexcept { return -static_cast<T>(b); }

	template<typename T, std::size_t N, std::size_t VSize>
	[[nodiscard]] constexpr std::size_t align_data() noexcept
	{
		const auto size_mult = VSize / sizeof(T);
		return N / size_mult + !!(N % size_mult);
	}
}