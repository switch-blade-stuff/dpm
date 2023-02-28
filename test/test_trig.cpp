/*
 * Created by switchblade on 2023-02-28.
 */

#include "common.hpp"

template<typename T, typename Abi>
static inline void test_trig() noexcept
{
	constexpr auto invoke_test = []<std::size_t N>(auto f, std::span<const T, N> vals)
	{
		constexpr auto min_err = std::same_as<T, float> ? T{2.0e-6} : T{6.0e-17};
		constexpr auto max_err = std::same_as<T, float> ? T{1.0e-3} : T{2.2e-16};
		constexpr auto simd_size = dpm::simd_size_v<T, Abi>;
		for (std::size_t i = 0; i < N;)
		{
			dpm::simd<T, Abi> v = {};
			v.copy_from(vals.data() + i, dpm::element_aligned);
			v = f(v);

			for (std::size_t j = 0; i < N && j < simd_size; ++j, ++i)
			{
				const auto s = f(vals[i]);
				TEST_ASSERT(almost_equal(v[j], s, max_err, min_err) || (std::isnan(v[j]) && std::isnan(s)));
			}
		}
	};

	const auto test_vals = std::array{
			T{0.0}, T{-0.0}, T{1.0}, T{-1.0}, T{0.125}, T{-0.125}, T{0.1234}, T{-0.1234}, T{0.34}, T{-0.34},
			T{12.54}, T{-12.54}, T{12299.99}, T{-12299.99}, std::numbers::pi_v<T>,
			std::numbers::pi_v<T> * 2, std::numbers::pi_v<T> / 4,
			std::numbers::pi_v<T> * 4, std::numbers::pi_v<T> / 6,
			std::numbers::pi_v<T> * 3, std::numbers::pi_v<T> / 3,
			std::numbers::pi_v<T> * 5, std::numbers::pi_v<T> / 5,
			std::numbers::pi_v<T> * 10, std::numbers::pi_v<T> / 10,
			std::numbers::pi_v<T> * 20, std::numbers::pi_v<T> / 20,
			T{std::numeric_limits<T>::max()}, T{std::numeric_limits<T>::min()},
			std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
			std::numeric_limits<T>::quiet_NaN(),
	};


	using dpm::sin;
	using std::sin;
	invoke_test([](auto x) { return sin(x); }, std::span{test_vals});
	using dpm::cos;
	using std::cos;
	invoke_test([](auto x) { return cos(x); }, std::span{test_vals});
	using dpm::tan;
	using std::tan;
	invoke_test([](auto x) { return tan(x); }, std::span{test_vals});
	using dpm::asin;
	using std::asin;
	invoke_test([](auto x) { return asin(x); }, std::span{test_vals});
	using dpm::acos;
	using std::acos;
	invoke_test([](auto x) { return acos(x); }, std::span{test_vals});
	using dpm::atan;
	using std::atan;
	invoke_test([](auto x) { return atan(x); }, std::span{test_vals});

	/* TODO: If DPM_HANDLE_ERRORS is set and fp exceptions are used, check exceptions. */
}
template<typename T>
static void test_trig() noexcept
{
	test_trig<T, dpm::simd_abi::scalar>();
	test_trig<T, dpm::simd_abi::fixed_size<1>>();
	test_trig<T, dpm::simd_abi::fixed_size<2>>();
	test_trig<T, dpm::simd_abi::fixed_size<4>>();
	test_trig<T, dpm::simd_abi::fixed_size<8>>();
	test_trig<T, dpm::simd_abi::fixed_size<16>>();
	test_trig<T, dpm::simd_abi::fixed_size<20>>();
	test_trig<T, dpm::simd_abi::fixed_size<24>>();
	test_trig<T, dpm::simd_abi::fixed_size<28>>();
	test_trig<T, dpm::simd_abi::fixed_size<32>>();
}

int main()
{
	test_trig<float>();
	test_trig<double>();
}
