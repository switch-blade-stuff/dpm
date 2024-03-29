/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

auto foo() noexcept
{
	constexpr auto v = dpm::fixed_size_simd<float, 4>{2};
	return dpm::ilogb(v);
}

template<typename T, typename Abi>
static inline void test_nextafter() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto a_data = std::array<T, 8 + simd_size>{
			T{-0.0}, T{-0.0}, T{0.0}, T{1.0},
			std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
			std::numeric_limits<T>::quiet_NaN(), T{0.0}
	};
	const auto b_data = std::array<T, 8 + simd_size>{
			T{0.0}, T{1.0}, T{1.0}, T{2.0},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			T{0.0}, std::numeric_limits<T>::quiet_NaN()
	};

	for (std::size_t i = 0; i < a_data.size(); i += simd_size)
	{
		const auto a = dpm::simd<T, Abi>{a_data.data() + i, dpm::element_aligned};
		const auto b = dpm::simd<T, Abi>{b_data.data() + i, dpm::element_aligned};
		const auto c = dpm::nextafter(a, b);

		for (std::size_t j = 0; i + j < a_data.size() && j < simd_size; ++j)
		{
			const auto s = std::nextafter(a[j], b[j]);
			TEST_ASSERT(c[j] == s || (std::isnan(c[j]) && std::isnan(s)));
		}
	}
}
template<typename T, typename Abi>
static inline void test_ilogb() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto data = std::array<T, 32 + simd_size>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25}, T{3.5}, T{-3.5}, T{3.8},
			std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
	};

	for (std::size_t i = 0; i < data.size(); i += simd_size)
	{
		const auto x = dpm::simd<T, Abi>{data.data() + i, dpm::element_aligned};
		const auto y = dpm::ilogb(x);

		for (std::size_t j = 0; i + j < data.size() && j < simd_size; ++j)
		{
			const auto s = std::ilogb(data[i + j]);
			TEST_ASSERT(y[j] == s);
		}
	}
}
template<typename T, typename Abi>
static inline void test_frexp() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto data = std::array<T, 32 + simd_size>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
			std::numeric_limits<T>::quiet_NaN(),
	};

	for (std::size_t i = 0; i < data.size(); i += simd_size)
	{
		auto e = dpm::simd<int, Abi>{};
		const auto x = dpm::simd<T, Abi>{data.data() + i, dpm::element_aligned};
		const auto y = dpm::frexp(x, e);

		for (std::size_t j = 0; i + j < data.size() && j < simd_size; ++j)
		{
			int se;
			const auto s = std::frexp(data[i + j], &se);

			TEST_ASSERT((std::isinf(y[j]) && std::isinf(s)) || (std::isnan(y[j]) && std::isnan(s)) ||
						(e[j] == se && almost_equal(y[j], s, std::numeric_limits<T>::epsilon())));
		}
	}
}
template<typename T, typename Abi>
static inline void test_ldexp() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto data = std::array<T, 32 + simd_size>{
			T{0}, T{-0}, T{0.1}, T{-0.1}, T{0.25}, T{-0.25}, T{0.5}, T{-0.5}, T{0.8}, T{-0.8},
			T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
			std::numeric_limits<T>::quiet_NaN(),
	};
	const auto exp_vals = std::array{0, 1, -1, 2, -2, 0xff, -0xff, 0x7ff, -0x7ff, std::numeric_limits<int>::max(), std::numeric_limits<int>::lowest(),};

	for (std::size_t i = 0; i < data.size(); i += simd_size)
	{
		const auto x = dpm::simd<T, Abi>{data.data() + i, dpm::element_aligned};
		for (const auto e: exp_vals)
		{
			const auto y = dpm::ldexp(x, e);
			for (std::size_t j = 0; i + j < data.size() && j < simd_size; ++j)
			{
				const auto s = std::ldexp(data[i + j], e);
				TEST_ASSERT(almost_equal(y[j], s, std::numeric_limits<T>::epsilon()));
			}
		}
	}
}
template<typename T, typename Abi>
static inline void test_modf() noexcept
{
	constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
	const auto data = std::array<T, 32 + simd_size>{
			T{0}, T{1}, T{-1}, T{1.1}, T{-1.1}, T{1.25}, T{-1.25}, T{1.5}, T{-1.5}, T{1.8}, T{-1.8},
			T{2}, T{-2}, T{2.1}, T{-2.1}, T{2.25}, T{-2.25}, T{2.5}, T{-2.5}, T{2.8}, T{-2.8},
			T{3}, T{-3}, T{3.1}, T{-3.1}, T{3.25}, T{-3.25},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(),
			std::numeric_limits<T>::quiet_NaN(),
	};

	for (std::size_t i = 0; i < data.size(); i += simd_size)
	{
		auto vi = dpm::simd<T, Abi>{};
		const auto x = dpm::simd<T, Abi>{data.data() + i, dpm::element_aligned};
		const auto vf = dpm::modf(x, vi);

		for (std::size_t j = 0; i + j < data.size() && j < simd_size; ++j)
		{
			T si;
			const auto sf = std::modf(data[i + j], &si);

			TEST_ASSERT(almost_equal(vf[j], sf, std::numeric_limits<T>::epsilon()) &&
			            almost_equal(vi[j], si, std::numeric_limits<T>::epsilon()));
		}
	}
}

template<typename T, typename Abi>
static void test_fmanip() noexcept
{
	test_nextafter<T, Abi>();
	test_ilogb<T, Abi>();
	test_frexp<T, Abi>();
	test_ldexp<T, Abi>();
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
