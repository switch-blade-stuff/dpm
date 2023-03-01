/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

template<typename T, typename Abi>
static inline void test_frexp() noexcept
{
	const auto test_vals = std::array<T, 32>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
			std::numeric_limits<T>::quiet_NaN(),
	};

	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	for (std::size_t i = 0; i < test_vals.size();)
	{
		auto e = dpm::simd<int, Abi>{};
		const auto x = dpm::simd<T, Abi>{test_vals.data() + i, dpm::element_aligned};
		const auto y = dpm::frexp(x, e);

		for (std::size_t j = 0; i < test_vals.size() && j < simd_size; ++j, ++i)
		{
			int se;
			const auto s = std::frexp(test_vals[i], &se);
			TEST_ASSERT((e[j] == se && almost_equal(y[j], s, std::numeric_limits<T>::epsilon())) ||
			            (std::isinf(y[j]) && std::isinf(s)) ||
			            (std::isnan(y[j]) && std::isnan(s)));
		}
	}

	/* TODO: If DPM_HANDLE_ERRORS is set and fp exceptions are used, check exceptions. */
}
template<typename T>
static void test_frexp() noexcept
{
	test_frexp<T, dpm::simd_abi::scalar>();
	test_frexp<T, dpm::simd_abi::fixed_size<1>>();
	test_frexp<T, dpm::simd_abi::fixed_size<2>>();
	test_frexp<T, dpm::simd_abi::fixed_size<4>>();
	test_frexp<T, dpm::simd_abi::fixed_size<8>>();
	test_frexp<T, dpm::simd_abi::fixed_size<16>>();
	test_frexp<T, dpm::simd_abi::fixed_size<20>>();
	test_frexp<T, dpm::simd_abi::fixed_size<24>>();
	test_frexp<T, dpm::simd_abi::fixed_size<28>>();
	test_frexp<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_frexp<float>();
	test_frexp<double>();
}
