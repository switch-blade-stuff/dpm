/*
 * Created by switchblade on 2023-03-03.
 */

#include "common.hpp"

template<typename T, typename Abi>
static inline void test_log() noexcept
{
	constexpr auto invoke_test = [](auto f)
	{
		constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
		const auto test_vals = std::array<T, 10 + simd_size>{
				T{12.54}, T{0.1234}, T{-0.34}, T{0.0}, T{1.0}, T{-1.0}, T{0.125},
				std::numbers::pi_v<T>, std::numeric_limits<T>::quiet_NaN(),
				std::numeric_limits<T>::infinity()
		};
		
		for (std::size_t i = 0; i < test_vals.size() - simd_size;)
		{
			const auto v = f(dpm::simd<T, Abi>{test_vals.data() + i, dpm::element_aligned});
			for (std::size_t j = 0; i < test_vals.size() - simd_size && j < simd_size; ++j, ++i)
			{
				const auto s = f(test_vals[i]);
				TEST_ASSERT(almost_equal(v[j], s, T{1.0e-3}, T{1.0e-7}));
			}
		}
	};

	using dpm::log;
	using std::log;
	invoke_test([](auto x) { return log(x); });
	using dpm::log2;
	using std::log2;
	invoke_test([](auto x) { return log2(x); });
	using dpm::log10;
	using std::log10;
	invoke_test([](auto x) { return log10(x); });
	using dpm::log1p;
	using std::log1p;
	invoke_test([](auto x) { return log1p(x); });
}

template<typename T>
static void test_expon() noexcept
{
	test_log<T, dpm::simd_abi::scalar>();
	test_log<T, dpm::simd_abi::fixed_size<1>>();
	test_log<T, dpm::simd_abi::fixed_size<2>>();
	test_log<T, dpm::simd_abi::fixed_size<4>>();
	test_log<T, dpm::simd_abi::fixed_size<8>>();
	test_log<T, dpm::simd_abi::fixed_size<16>>();
	test_log<T, dpm::simd_abi::fixed_size<20>>();
	test_log<T, dpm::simd_abi::fixed_size<24>>();
	test_log<T, dpm::simd_abi::fixed_size<28>>();
	test_log<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_expon<float>();
	test_expon<double>();
}
