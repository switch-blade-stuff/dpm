/*
 * Created by switchblade on 2023-01-01.
 */

#define DPM_DEBUG

#include <dpm/simd.hpp>

#define TEST_ASSERT(x) assert(x)

template<typename T, typename Abi, typename mask_t = dpm::simd_mask<T, Abi>>
static inline void test_mask() noexcept
{
	mask_t a = {true}, b = {true}, c = {false}, d = {false};

	TEST_ASSERT(dpm::all_of(a));
	TEST_ASSERT(!dpm::all_of(c));

	TEST_ASSERT(dpm::any_of(a));
	TEST_ASSERT(!dpm::any_of(c));

	TEST_ASSERT(dpm::none_of(c));
	TEST_ASSERT(!dpm::none_of(a));

	TEST_ASSERT(dpm::all_of(a == b));
	TEST_ASSERT(dpm::all_of(c == d));

	TEST_ASSERT(dpm::all_of((!c) == a));
	TEST_ASSERT(dpm::all_of((!a) == c));

	TEST_ASSERT(dpm::none_of(a != b));
	TEST_ASSERT(dpm::none_of(c != d));

	TEST_ASSERT(dpm::none_of(a == c));
	TEST_ASSERT(dpm::none_of(a == d));

	TEST_ASSERT(dpm::all_of(b || c));
	TEST_ASSERT(dpm::none_of(b && c));

	TEST_ASSERT(dpm::all_of(b | c));
	TEST_ASSERT(dpm::none_of(c | d));

	TEST_ASSERT(dpm::all_of(a & b));
	TEST_ASSERT(dpm::none_of(b & c));
	TEST_ASSERT(dpm::none_of(c & d));

	TEST_ASSERT(dpm::all_of(b ^ c));
	TEST_ASSERT(dpm::all_of(c ^ b));
	TEST_ASSERT(dpm::none_of(a ^ b));
	TEST_ASSERT(dpm::none_of(c ^ d));

	if constexpr (mask_t::size() > 1)
	{
		std::array<bool, mask_t::size()> a_vals;
		for (std::size_t i = 0; i < mask_t::size(); ++i) a_vals[i] = i % 2;
		std::array<bool, mask_t::size()> b_vals;
		for (std::size_t i = 0; i < mask_t::size(); ++i) b_vals[i] = false;
		std::array<bool, mask_t::size()> c_vals;
		for (std::size_t i = 0; i < mask_t::size(); ++i) c_vals[i] = true;

		a.copy_from(a_vals.data(), dpm::element_aligned);
		b.copy_from(b_vals.data(), dpm::element_aligned);
		c.copy_from(c_vals.data(), dpm::element_aligned);

		TEST_ASSERT(dpm::some_of(a));
		TEST_ASSERT(!dpm::some_of(b));
		TEST_ASSERT(!dpm::some_of(c));

		TEST_ASSERT(dpm::popcount(a) == mask_t::size() / 2);
		TEST_ASSERT(dpm::popcount(b) == 0);
		TEST_ASSERT(dpm::popcount(c) == mask_t::size());

		TEST_ASSERT(dpm::find_first_set(a) == 1);
		TEST_ASSERT(dpm::find_first_set(c) == 0);
		TEST_ASSERT(dpm::find_last_set(a) == mask_t::size() - 1);
		TEST_ASSERT(dpm::find_last_set(c) == mask_t::size() - 1);

		std::array<bool, mask_t::size()> tmp = {};

		a.copy_to(tmp.data(), dpm::element_aligned);
		TEST_ASSERT(tmp == a_vals);

		b.copy_to(tmp.data(), dpm::element_aligned);
		TEST_ASSERT(tmp == b_vals);

		c.copy_to(tmp.data(), dpm::element_aligned);
		TEST_ASSERT(tmp == c_vals);
	}
}

#include <cmath>

int main()
{
	test_mask<float, dpm::simd_abi::scalar>();
	test_mask<float, dpm::simd_abi::fixed_size<4>>();
	test_mask<float, dpm::simd_abi::fixed_size<8>>();
	test_mask<float, dpm::simd_abi::fixed_size<16>>();
	test_mask<float, dpm::simd_abi::fixed_size<32>>();
	test_mask<float, dpm::simd_abi::ext::aligned_vector<4, 16>>();
	test_mask<float, dpm::simd_abi::ext::aligned_vector<8, 16>>();
	test_mask<float, dpm::simd_abi::ext::aligned_vector<16, 16>>();
	test_mask<float, dpm::simd_abi::ext::aligned_vector<32, 16>>();

	{
		dpm::simd<float, dpm::simd_abi::fixed_size<4>> a = {1.0f}, b = {-1.0f}, d = {0.0f};
		const auto c = -a;

		TEST_ASSERT(dpm::all_of(a != b));
		TEST_ASSERT(dpm::none_of(a == b));
		TEST_ASSERT(dpm::all_of(!(a == b)));

		TEST_ASSERT(dpm::all_of(c == b));
		TEST_ASSERT(dpm::none_of(c != b));
		TEST_ASSERT(dpm::none_of(!(c == b)));

		TEST_ASSERT(dpm::all_of(a + b == d));
		TEST_ASSERT(dpm::all_of(d - a == b));
		TEST_ASSERT(dpm::all_of(d - b == a));
		TEST_ASSERT(dpm::all_of(b + b == b * decltype(b){2.0f}));

		TEST_ASSERT(dpm::reduce(d) == 0.0f);
		TEST_ASSERT(dpm::reduce(a) == 1.0f * a.size());
		TEST_ASSERT(dpm::reduce(b) == -1.0f * a.size());

		const auto ia = dpm::static_simd_cast<std::int32_t>(a);
		const auto ib = dpm::static_simd_cast<std::int32_t>(b);
		const auto ic = dpm::static_simd_cast<std::int32_t>(c);
		const auto id = dpm::static_simd_cast<std::int32_t>(d);

		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(ia) == a));
		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(ib) == b));
		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(ic) == c));
		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(id) == d));

		TEST_ASSERT(dpm::all_of(ia != ib));
		TEST_ASSERT(dpm::none_of(ia == ib));
		TEST_ASSERT(dpm::all_of(!(ia == ib)));

		TEST_ASSERT(dpm::all_of(ic == ib));
		TEST_ASSERT(dpm::none_of(ic != ib));
		TEST_ASSERT(dpm::none_of(!(ic == ib)));

		TEST_ASSERT(dpm::all_of(ia + ib == id));
		TEST_ASSERT(dpm::all_of(id - ia == ib));
		TEST_ASSERT(dpm::all_of(id - ib == ia));
		TEST_ASSERT(dpm::all_of(ib + ib == ib * decltype(ib){2}));

		const auto da = dpm::simd_cast<double>(a);
		const auto db = dpm::simd_cast<double>(b);
		const auto dc = dpm::simd_cast<double>(c);
		const auto dd = dpm::simd_cast<double>(d);

		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(da) == a));
		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(db) == b));
		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(dc) == c));
		TEST_ASSERT(dpm::all_of(dpm::static_simd_cast<float>(dd) == d));

		const auto b2 = dpm::split_by<2>(b);
		const auto c2 = dpm::split_by<2>(c);

		TEST_ASSERT(dpm::all_of(b2[0] == b2[0]));
		TEST_ASSERT(dpm::all_of(b2[1] == b2[1]));
		TEST_ASSERT(dpm::all_of(b2[0] == c2[0]));
		TEST_ASSERT(dpm::all_of(b2[1] == c2[1]));

		TEST_ASSERT(dpm::all_of(decltype(b){dpm::concat(b2)} == b));
		TEST_ASSERT(dpm::all_of(decltype(b){dpm::concat(b2)} == c));
		TEST_ASSERT(dpm::all_of(decltype(b){dpm::concat(c2)} == c));

		const auto b4 = dpm::split_by<4>(b);
		const auto c4 = dpm::split_by<4>(c);

		TEST_ASSERT(dpm::all_of(b4[0] == b4[0]));
		TEST_ASSERT(dpm::all_of(b4[1] == b4[1]));
		TEST_ASSERT(dpm::all_of(b4[2] == b4[2]));
		TEST_ASSERT(dpm::all_of(b4[3] == b4[3]));
		TEST_ASSERT(dpm::all_of(b4[0] == c4[0]));
		TEST_ASSERT(dpm::all_of(b4[1] == c4[1]));
		TEST_ASSERT(dpm::all_of(b4[2] == c4[2]));
		TEST_ASSERT(dpm::all_of(b4[3] == c4[3]));

		TEST_ASSERT(dpm::all_of(decltype(b){dpm::concat(b4)} == b));
		TEST_ASSERT(dpm::all_of(decltype(b){dpm::concat(b4)} == c));
		TEST_ASSERT(dpm::all_of(decltype(b){dpm::concat(c4)} == c));

		alignas(decltype(d)) std::array<float, 4> d_vals;
		for (std::size_t i = 0; i < d_vals.size(); ++i) d_vals[i] = i % 2 ? b[i] : a[i];
		alignas(decltype(d)::mask_type) std::array<bool, 4> mask_vals;
		for (std::size_t i = 0; i < mask_vals.size(); ++i) mask_vals[i] = i % 2;

		typename decltype(d)::mask_type mask;
		d.copy_from(d_vals.data(), dpm::vector_aligned);
		mask.copy_from(mask_vals.data(), dpm::vector_aligned);

		TEST_ASSERT(dpm::all_of(dpm::ext::blend(a, where(mask, b)) == d));

		alignas(decltype(d)) std::array<float, 4> v_tmp;
		where(mask, b).copy_to(v_tmp.data(), dpm::vector_aligned);
		where(!mask, a).copy_to(v_tmp.data(), dpm::vector_aligned);

		TEST_ASSERT(v_tmp == d_vals);

		TEST_ASSERT(dpm::all_of(dpm::min(d, b) == b));
		TEST_ASSERT(dpm::all_of(dpm::max(d, a) == a));
		TEST_ASSERT(dpm::all_of(dpm::clamp(d, decltype(d){.5f}, decltype(d){.5f}) == decltype(b){0.5f}));

		const auto e = dpm::ext::shuffle<1, 0, 3, 2>(d);
		where(mask, a).copy_to(d_vals.data(), dpm::vector_aligned);
		where(!mask, b).copy_to(d_vals.data(), dpm::vector_aligned);
		d.copy_from(d_vals.data(), dpm::vector_aligned);

		TEST_ASSERT(dpm::all_of(d == e));
		TEST_ASSERT(dpm::all_of(dpm::ext::shuffle<0, 0, 0, 0>(d) == b));
		TEST_ASSERT(dpm::all_of(dpm::ext::shuffle<1, 1, 1, 1>(d) == a));
		TEST_ASSERT(dpm::all_of(dpm::ext::shuffle<0, 0, 0, 0>(d) == dpm::ext::shuffle<0, 0, 0, 0>(d)));
		TEST_ASSERT(dpm::all_of(dpm::ext::shuffle<1, 1, 1, 1>(d) == dpm::ext::shuffle<3, 3, 3, 3>(d)));

		const auto f = dpm::concat(dpm::ext::shuffle<0>(b), dpm::ext::shuffle<1>(a));
		TEST_ASSERT(dpm::all_of(dpm::ext::shuffle<0, 1>(d) == f));
	}
	{
		std::array<float, 3> data = {-1, 2, -3};
		dpm::fixed_size_simd<float, 3> a;
		a.copy_from(data.data(), dpm::element_aligned);

		TEST_ASSERT(dpm::reduce(a, std::plus<>{}) == (-1 + 2 - 3));
		TEST_ASSERT(dpm::reduce(a, std::multiplies<>{}) == (-1 * 2 * -3));
	}
	{
		dpm::simd<std::int64_t, dpm::simd_abi::fixed_size<2>> a = {1}, b = {2}, c = {4}, d = {~4ll};

		TEST_ASSERT(dpm::all_of(a != b));
		TEST_ASSERT(dpm::none_of(a == b));
		TEST_ASSERT(dpm::all_of(!(a == b)));

		TEST_ASSERT(dpm::all_of((a << b) == c));
		TEST_ASSERT(dpm::all_of((c >> b) == a));

		TEST_ASSERT(dpm::all_of(~c == d));
		TEST_ASSERT(dpm::all_of(!c == !!decltype(c){}));
	}
	{
		dpm::simd<double, dpm::simd_abi::fixed_size<2>> a = {0.1234}, b = {12.7};
		dpm::simd<double, dpm::simd_abi::scalar> c = {0.1234}, d = {12.7};

		TEST_ASSERT(std::abs(dpm::sin(a)[0] - dpm::sin(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::sin(a)[1] - dpm::sin(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::sin(b)[0] - dpm::sin(d)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::sin(b)[1] - dpm::sin(d)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(a)[0] - dpm::cos(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(a)[1] - dpm::cos(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(b)[0] - dpm::cos(d)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(b)[1] - dpm::cos(d)[0]) < 0.0001);
	}
	{
		dpm::simd<double, dpm::simd_abi::fixed_size<4>> a = {0.1234}, b = {12.7};
		dpm::simd<double, dpm::simd_abi::scalar> c = {0.1234}, d = {12.7};

		TEST_ASSERT(std::abs(dpm::sin(a)[0] - dpm::sin(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::sin(a)[1] - dpm::sin(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::sin(b)[0] - dpm::sin(d)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::sin(b)[1] - dpm::sin(d)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(a)[0] - dpm::cos(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(a)[1] - dpm::cos(c)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(b)[0] - dpm::cos(d)[0]) < 0.0001);
		TEST_ASSERT(std::abs(dpm::cos(b)[1] - dpm::cos(d)[0]) < 0.0001);
	}
	{
		const std::array<std::int16_t, 16> a_data = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		dpm::fixed_size_simd<std::int16_t, 16> a;
		a.copy_from(a_data.data(), dpm::element_aligned);

		const std::array<std::int16_t, 8> b_data = {1, 1, 2, 2, 4, 5, 4, 5};
		dpm::fixed_size_simd<std::int16_t, 8> b;
		b.copy_from(b_data.data(), dpm::element_aligned);

		const auto c = dpm::ext::shuffle<1, 1, 2, 2, 4, 5, 4, 5>(a);
		TEST_ASSERT(dpm::all_of(b == c));
	}
	{
		alignas(32) const std::array<std::int32_t, 8> a_data = {9, 10, 11, 12, 13, 14, 15};
		alignas(32) const std::array<bool, 8> m_data = {true, false, false, true, true};

		dpm::fixed_size_simd<std::int32_t, 8> a;
		a.copy_from(a_data.data(), dpm::vector_aligned);
		dpm::fixed_size_simd_mask<std::int32_t, 8> m;
		m.copy_from(m_data.data(), dpm::vector_aligned);

		alignas(32) const std::array<std::int32_t, 8> b_data = {0, 10, 11, 3, 4, 14, 15};
		alignas(32) const std::array<std::int32_t, 8> c_data = {0, 1, 2, 3, 4, 5, 6, 7};
		dpm::fixed_size_simd<std::int32_t, 8> b;
		b.copy_from(b_data.data(), dpm::vector_aligned);

		dpm::where(m, a).copy_from(c_data.data(), dpm::vector_aligned);
		TEST_ASSERT(dpm::all_of(a == b));
	}
}
