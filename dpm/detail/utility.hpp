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
	template<std::size_t N>
	[[nodiscard]] constexpr std::size_t fill_bits() noexcept
	{
		std::size_t result = 0;
		for (std::size_t i = 0; i < N; ++i)
			result |= 1ull << i;
		return result;
	}

	template<typename T>
	[[nodiscard]] constexpr T extend_bool(bool b) noexcept { return -static_cast<T>(b); }

	template<typename T, std::size_t N, std::size_t VSize>
	[[nodiscard]] constexpr std::size_t align_data() noexcept
	{
		const auto size_mult = VSize / sizeof(T);
		return N / size_mult + !!(N % size_mult);
	}

	template<typename, std::size_t...>
	struct reverse_sequence;
	template<std::size_t I, std::size_t... Js>
	struct reverse_sequence<std::index_sequence<Js...>, I> : std::index_sequence<Js..., I> {};
	template<std::size_t I, std::size_t... Is, std::size_t... Js>
	struct reverse_sequence<std::index_sequence<Js...>, I, Is...> : reverse_sequence<std::index_sequence<Js..., I>, Is...> {};
	template<std::size_t... Is>
	using reverse_sequence_t = reverse_sequence<std::index_sequence<>, Is...>;

	template<typename, typename, std::size_t, std::size_t, std::size_t...>
	struct extract_sequence;
	template<std::size_t P, std::size_t N, std::size_t I, std::size_t... Is, std::size_t... Js, std::size_t... Ks>
	struct extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks...>, P, N, I, Is...> : extract_sequence<std::index_sequence<I, Js...>, std::index_sequence<Ks...>, P, N, Is...> {};
	template<std::size_t P, std::size_t N, std::size_t I, std::size_t... Is, std::size_t... Js, std::size_t... Ks> requires (sizeof...(Js) == P)
	struct extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks...>, P, N, I, Is...> : extract_sequence<std::index_sequence<Js...>, std::index_sequence<I, Ks...>, P, N, Is...> {};
	template<std::size_t P, std::size_t N, std::size_t... Is, std::size_t... Js, std::size_t... Ks> requires (sizeof...(Js) == P && sizeof...(Ks) == N)
	struct extract_sequence<std::index_sequence<Js...>, std::index_sequence<Ks...>, P, N, Is...> : std::index_sequence<Ks...> {};
	template<std::size_t Pos, std::size_t N, std::size_t... Is>
	using extract_sequence_t = extract_sequence<std::index_sequence<>, std::index_sequence<>, Pos, N, Is...>;

	template<typename, std::size_t, std::size_t, std::size_t...>
	struct repeat_sequence;
	template<std::size_t N, std::size_t I, std::size_t... Js>
	struct repeat_sequence<std::index_sequence<Js...>, N, N, I> : std::index_sequence<Js...> {};
	template<std::size_t N, std::size_t I, std::size_t... Is, std::size_t... Js>
	struct repeat_sequence<std::index_sequence<Js...>, N, N, I, Is...> : reverse_sequence<std::index_sequence<Js...>, N, 0, Is...> {};
	template<std::size_t N, std::size_t J, std::size_t I, std::size_t... Is, std::size_t... Js>
	struct repeat_sequence<std::index_sequence<Js...>, N, J, I, Is...> : reverse_sequence<std::index_sequence<Js..., I>, N, J + 1, I, Is...> {};
	template<std::size_t N, std::size_t... Is>
	using repeat_sequence_t = repeat_sequence<std::index_sequence<>, N, 0, Is...>;

	template<typename, std::size_t, std::size_t, std::size_t...>
	struct pad_sequence;
	template<std::size_t N, std::size_t Inc, std::size_t I, std::size_t... Js> requires (sizeof...(Js) >= N)
	struct pad_sequence<std::index_sequence<Js...>, N, Inc, I> : std::index_sequence<Js...> {};
	template<std::size_t N, std::size_t Inc, std::size_t I, std::size_t... Js>
	struct pad_sequence<std::index_sequence<Js...>, N, Inc, I> : pad_sequence<std::index_sequence<Js..., I>, N, Inc, I + Inc> {};
	template<std::size_t N, std::size_t Inc, std::size_t I, std::size_t... Is, std::size_t... Js>
	struct pad_sequence<std::index_sequence<Js...>, N, Inc, I, Is...> : pad_sequence<std::index_sequence<Js..., I>, N, Inc, Is...> {};
	template<std::size_t N, std::size_t Inc, std::size_t... Is>
	using pad_sequence_t = pad_sequence<std::index_sequence<>, N, Inc, Is...>;

	template<std::size_t I, std::size_t... Is>
	constexpr void copy_positions(std::index_sequence<Is...>, auto *dst, const auto *src) noexcept
	{
		*dst = src[I];
		if constexpr (sizeof...(Is) != 0) copy_positions<Is...>(dst + 1, src);
	}
}