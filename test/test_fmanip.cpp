/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

#include <cstdio>


template<typename T, typename Abi>
static inline void test_ilogb() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto test_vals = std::array<T, 32 + simd_size>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25}, T{3.5}, T{-3.5}, T{3.8},
			std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
	};

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
template<typename T, typename Abi>
static inline void test_frexp() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto test_vals = std::array<T, 32 + simd_size>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
			std::numeric_limits<T>::quiet_NaN(),
	};

	for (std::size_t i = 0; i < test_vals.size() - simd_size;)
	{
		auto e = dpm::simd<int, Abi>{};
		const auto x = dpm::simd<T, Abi>{test_vals.data() + i, dpm::element_aligned};
		const auto y = dpm::frexp(x, e);

		for (std::size_t j = 0; i < test_vals.size() - simd_size && j < simd_size; ++j, ++i)
		{
			int se;
			const auto s = std::frexp(test_vals[i], &se);

			TEST_ASSERT((e[j] == se && almost_equal(y[j], s, std::numeric_limits<T>::epsilon())) ||
			            (std::isinf(y[j]) && std::isinf(s)) ||
			            (std::isnan(y[j]) && std::isnan(s)));
		}
	}
}
template<typename T, typename Abi>
static inline void test_modf() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto test_vals = std::array<T, 32 + simd_size>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::max(), std::numeric_limits<T>::min(),
			std::numeric_limits<T>::quiet_NaN(),
	};

	for (std::size_t i = 0; i < test_vals.size() - simd_size;)
	{
		auto vi = dpm::simd<T, Abi>{};
		const auto x = dpm::simd<T, Abi>{test_vals.data() + i, dpm::element_aligned};
		const auto vf = dpm::modf(x, vi);

		for (std::size_t j = 0; i < test_vals.size() - simd_size && j < simd_size; ++j, ++i)
		{
			T si;
			const auto sf = std::modf(test_vals[i], &si);

			TEST_ASSERT(almost_equal(vf[j], sf, std::numeric_limits<T>::epsilon()) &&
			            almost_equal(vi[j], si, std::numeric_limits<T>::epsilon()));
		}
	}
}

template<typename T, typename Abi>
static void test_fmanip() noexcept
{
	test_ilogb<T, Abi>();
	test_frexp<T, Abi>();
	test_modf<T, Abi>();
}
template<typename T>
static void test_fmanip() noexcept
{
	test_fmanip<T, dpm::simd_abi::scalar>();
	test_fmanip<T, dpm::simd_abi::fixed_size<1>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<2>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<4>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<8>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<16>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<20>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<24>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<28>>();
	test_fmanip<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_fmanip<float>();
	test_fmanip<double>();
}
