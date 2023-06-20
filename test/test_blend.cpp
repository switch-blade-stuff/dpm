/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

template<typename T, typename Abi, std::size_t... Is>
static void test_mask_blend(std::index_sequence<Is...>) noexcept
{
	using mask_t = dpm::simd_mask<T, Abi>;

	auto a_data = std::array<bool, sizeof...(Is)>{};
	auto b_data = std::array<bool, sizeof...(Is)>{};
	const auto m_data = std::array{(!!(Is % 2))...};
	std::fill_n(a_data.begin(), a_data.size(), false);
	std::fill_n(b_data.begin(), b_data.size(), true);
	const auto a = mask_t{a_data.data(), dpm::element_aligned};
	const auto b = mask_t{b_data.data(), dpm::element_aligned};
	const auto m = mask_t{m_data.data(), dpm::element_aligned};
	const auto c = dpm::ext::blend(a, b, m);

	for (std::size_t i = 0; i < a.size(); ++i)
		TEST_ASSERT(c[i] == _data[i]);
}
template<typename T, typename Abi>
static void test_mask_blend() noexcept{test_mask_blend<T, Abi>(std::make_index_sequence<dpm::simd_size_v<T, Abi>>{});}
template<typename T>
static void test_mask_blend() noexcept
{
	test_mask_blend<T, dpm::simd_abi::scalar>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<1>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<2>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<4>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<8>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<16>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<20>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<24>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<28>>();
	test_mask_blend<T, dpm::simd_abi::fixed_size<32>>();
}

template<typename T, typename Abi, std::size_t... Is>
static void test_simd_blend(std::index_sequence<Is...>) noexcept
{
	using mask_t = dpm::simd_mask<T, Abi>;
	using simd_t = dpm::simd<T, Abi>;

	auto a_data = std::array<T, sizeof...(Is)>{};
	auto b_data = std::array<T, sizeof...(Is)>{};
	const auto c_data = std::array{(!!(Is % 2) ? T{1} : T{0})...};
	const auto m_data = std::array{(!!(Is % 2))...};
	std::fill_n(a_data.begin(), a_data.size(), static_cast<T>(0));
	std::fill_n(b_data.begin(), b_data.size(), static_cast<T>(1));
	const auto a = simd_t{a_data.data(), dpm::element_aligned};
	const auto b = simd_t{b_data.data(), dpm::element_aligned};
	const auto m = mask_t{m_data.data(), dpm::element_aligned};
	const auto c = dpm::ext::blend(a, b, m);

	for (std::size_t i = 0; i < a.size(); ++i)
		TEST_ASSERT(c[i] == c_data[i]);
}
template<typename T, typename Abi>
static void test_simd_blend() noexcept{test_simd_blend<T, Abi>(std::make_index_sequence<dpm::simd_size_v<T, Abi>>{});}
template<typename T>
static void test_simd_blend() noexcept
{
	test_simd_blend<T, dpm::simd_abi::scalar>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<1>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<2>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<4>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<8>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<16>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<20>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<24>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<28>>();
	test_simd_blend<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_mask_blend<std::int8_t>();
	test_simd_blend<std::int8_t>();
	test_mask_blend<std::int16_t>();
	test_simd_blend<std::int16_t>();
	test_mask_blend<std::int32_t>();
	test_simd_blend<std::int32_t>();
	test_mask_blend<std::int64_t>();
	test_simd_blend<std::int64_t>();
	test_mask_blend<std::uint8_t>();
	test_simd_blend<std::uint8_t>();
	test_mask_blend<std::uint16_t>();
	test_simd_blend<std::uint16_t>();
	test_mask_blend<std::uint32_t>();
	test_simd_blend<std::uint32_t>();
	test_mask_blend<std::uint64_t>();
	test_simd_blend<std::uint64_t>();
	test_mask_blend<float>();
	test_simd_blend<float>();
	test_mask_blend<double>();
	test_simd_blend<double>();
}