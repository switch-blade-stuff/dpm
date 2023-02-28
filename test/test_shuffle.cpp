/*
 * Created by switchblade on 2023-02-27.
 */

#include "common.hpp"

template<std::size_t, typename...>
struct shift_sequence;
template<std::size_t N, std::size_t I, std::size_t... Is>
struct shift_sequence<N, std::index_sequence<I, Is...>> : shift_sequence<N - 1, std::index_sequence<Is..., I>> {};
template<std::size_t I, std::size_t... Is>
struct shift_sequence<0, std::index_sequence<I, Is...>> { using type = std::index_sequence<I, Is...>; };

template<typename T, typename Abi, std::size_t I, std::size_t... Is, std::size_t... Js, std::size_t... Ks>
static void test_mask_shuffle(std::index_sequence<I, Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) noexcept
{
	using mask_t = dpm::simd_mask<T, Abi>;

	mask_t a, b;
	const auto a_data = std::array<bool, sizeof...(Js)>{!(Js % 2)...};
	const auto b_data = std::array<bool, sizeof...(Ks)>{!(Ks % 2)...};
	a.copy_from(a_data.data(), dpm::element_aligned);
	b.copy_from(b_data.data(), dpm::element_aligned);
	const auto c = dpm::ext::shuffle<Ks...>(a);

	for (std::size_t i = 0; i < sizeof...(Js); ++i)
		TEST_ASSERT(c[i] == b[i]);
}
template<typename T, typename Abi, std::size_t I, std::size_t... Is, std::size_t... Js>
static void test_mask_shuffle(std::index_sequence<I, Is...>, std::index_sequence<Js...>) noexcept
{
	test_mask_shuffle<T, Abi>(std::index_sequence<I, Is...>{}, std::index_sequence<Js...>{}, typename shift_sequence<I, std::index_sequence<Js...>>::type{});
	if constexpr (sizeof...(Is) != 0) test_mask_shuffle<T, Abi>(std::index_sequence<Is...>{}, std::index_sequence<Js...>{});
}
template<typename T, typename Abi>
static void test_mask_shuffle() noexcept
{
	test_mask_shuffle<T, Abi>(std::make_index_sequence<dpm::simd_size_v<T, Abi>>{},
	                          std::make_index_sequence<dpm::simd_size_v<T, Abi>>{});
}
template<typename T>
static void test_mask_shuffle() noexcept
{
	test_mask_shuffle<T, dpm::simd_abi::scalar>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<1>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<2>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<4>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<8>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<16>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<20>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<24>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<28>>();
	test_mask_shuffle<T, dpm::simd_abi::fixed_size<32>>();
}

template<typename T, typename Abi, std::size_t I, std::size_t... Is, std::size_t... Js>
static void test_simd_shuffle(std::index_sequence<I, Is...>, std::index_sequence<Js...>) noexcept
{
	using simd_t = dpm::simd<T, Abi>;

	const auto a_data = std::array{static_cast<T>(Js)...};
	const auto b_data = std::array{static_cast<T>(I), static_cast<T>(Is)...};
	const auto a = simd_t{a_data.data(), dpm::element_aligned};
	const auto b = dpm::ext::shuffle<I, Is...>(a);

	for (std::size_t i = 0; i < a.size(); ++i)
		TEST_ASSERT(b[i] == b_data[i]);

	if constexpr (I != sizeof...(Is)) test_simd_shuffle<T, Abi>(std::index_sequence<Is..., I>{}, std::index_sequence<Js...>{});
}
template<typename T, typename Abi>
static void test_simd_shuffle() noexcept
{
	test_simd_shuffle<T, Abi>(std::make_index_sequence<dpm::simd_size_v<T, Abi>>{},
	                          std::make_index_sequence<dpm::simd_size_v<T, Abi>>{});
}
template<typename T>
static void test_simd_shuffle() noexcept
{
	test_simd_shuffle<T, dpm::simd_abi::scalar>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<1>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<2>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<4>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<8>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<16>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<20>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<24>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<28>>();
	test_simd_shuffle<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_mask_shuffle<std::int8_t>();
	test_simd_shuffle<std::int8_t>();
	test_mask_shuffle<std::int16_t>();
	test_simd_shuffle<std::int16_t>();
	test_mask_shuffle<std::int32_t>();
	test_simd_shuffle<std::int32_t>();
	test_mask_shuffle<std::int64_t>();
	test_simd_shuffle<std::int64_t>();
	test_mask_shuffle<std::uint8_t>();
	test_simd_shuffle<std::uint8_t>();
	test_mask_shuffle<std::uint16_t>();
	test_simd_shuffle<std::uint16_t>();
	test_mask_shuffle<std::uint32_t>();
	test_simd_shuffle<std::uint32_t>();
	test_mask_shuffle<std::uint64_t>();
	test_simd_shuffle<std::uint64_t>();
	test_mask_shuffle<float>();
	test_simd_shuffle<float>();
	test_mask_shuffle<double>();
	test_simd_shuffle<double>();
}