/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

template<typename T, typename Abi>
static inline void test_ilogb() noexcept
{
	const auto test_vals = std::array<T, 32>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25}, T{3.5}, T{-3.5}, T{3.8},
			std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
	};

	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	for (std::size_t i = 0; i < test_vals.size();)
	{
		const auto x = dpm::simd<T, Abi>{test_vals.data() + i, dpm::element_aligned};
		const auto y = dpm::ilogb(x);

		for (std::size_t j = 0; i < test_vals.size() && j < simd_size; ++j, ++i)
		{
			const auto s = std::ilogb(test_vals[i]);
			TEST_ASSERT(y[j] == s);
		}
	}
}
template<typename T>
static void test_ilogb() noexcept
{
	test_ilogb<T, dpm::simd_abi::scalar>();
	test_ilogb<T, dpm::simd_abi::fixed_size<1>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<2>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<4>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<8>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<16>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<20>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<24>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<28>>();
	test_ilogb<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_ilogb<float>();
	test_ilogb<double>();
}
