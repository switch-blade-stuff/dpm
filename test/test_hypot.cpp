/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

template<typename T, typename Abi>
static inline void test_hypot() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto a_data = std::array<T, 10 + simd_size>{
			T{-0.0}, T{-0.0}, T{0.0}, T{0.0}, T{1.0}, T{2.0},
			std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
			std::numeric_limits<T>::infinity(), std::numeric_limits<T>::quiet_NaN()
	};
	const auto b_data = std::array<T, 10 + simd_size>{
			T{0.0}, T{1.0}, T{3.0}, T{-3.0}, T{2.0}, T{2.0},
			std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN(),
			std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()
	};

	for (std::size_t i = 0; i < a_data.size() - simd_size;)
	{
		const auto a = dpm::simd<T, Abi>{a_data.data() + i, dpm::element_aligned};
		const auto b = dpm::simd<T, Abi>{b_data.data() + i, dpm::element_aligned};
		const auto c0 = dpm::hypot(a, b);
		const auto c1 = dpm::hypot(b, a);
		const auto c2 = dpm::hypot(a, -b);

		for (std::size_t j = 0; i < a_data.size() - simd_size && j < simd_size; ++j, ++i)
		{
			const auto s0 = std::hypot(a[j], b[j]);
			const auto s1 = std::hypot(b[j], a[j]);
			const auto s2 = std::hypot(a[j], -b[j]);
			TEST_ASSERT(almost_equal(c0[j], s0));
			TEST_ASSERT(almost_equal(c1[j], s1));
			TEST_ASSERT(almost_equal(c2[j], s2));
		}
		TEST_ASSERT(dpm::all_of((c0 == c1 & c1 == c2) | (dpm::isnan(c0) & dpm::isnan(c1) & dpm::isnan(c2))));
	}
}

template<typename T>
static void test_hypot() noexcept
{
	test_hypot<T, dpm::simd_abi::scalar>();
	test_hypot<T, dpm::simd_abi::fixed_size<1>>();
	test_hypot<T, dpm::simd_abi::fixed_size<2>>();
	test_hypot<T, dpm::simd_abi::fixed_size<4>>();
	test_hypot<T, dpm::simd_abi::fixed_size<8>>();
	test_hypot<T, dpm::simd_abi::fixed_size<16>>();
	test_hypot<T, dpm::simd_abi::fixed_size<20>>();
	test_hypot<T, dpm::simd_abi::fixed_size<24>>();
	test_hypot<T, dpm::simd_abi::fixed_size<28>>();
	test_hypot<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_hypot<float>();
	test_hypot<double>();
}
