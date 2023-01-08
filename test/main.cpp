/*
 * Created by switchblade on 2023-01-01.
 */

#define SVM_DEBUG

#include <svm/type.hpp>

#define TEST_ASSERT(x) assert(x)

template<typename T, typename Abi, typename mask_t = svm::simd_mask<T, Abi>>
static inline void test_mask() noexcept
{
	mask_t a = {true}, b = {true}, c = {false}, d = {false};

	TEST_ASSERT(svm::all_of(a));
	TEST_ASSERT(!svm::all_of(c));

	TEST_ASSERT(svm::any_of(a));
	TEST_ASSERT(!svm::any_of(c));

	TEST_ASSERT(svm::none_of(c));
	TEST_ASSERT(!svm::none_of(a));

	TEST_ASSERT(svm::all_of(a == b));
	TEST_ASSERT(svm::all_of(c == d));

	TEST_ASSERT(svm::all_of((!c) == a));
	TEST_ASSERT(svm::all_of((!a) == c));

	TEST_ASSERT(svm::none_of(a != b));
	TEST_ASSERT(svm::none_of(c != d));

	TEST_ASSERT(svm::none_of(a == c));
	TEST_ASSERT(svm::none_of(a == d));

	TEST_ASSERT(svm::all_of(b || c));
	TEST_ASSERT(svm::none_of(b && c));

	TEST_ASSERT(svm::all_of(b | c));
	TEST_ASSERT(svm::none_of(c | d));

	TEST_ASSERT(svm::all_of(a & b));
	TEST_ASSERT(svm::none_of(b & c));
	TEST_ASSERT(svm::none_of(c & d));

	TEST_ASSERT(svm::all_of(b ^ c));
	TEST_ASSERT(svm::all_of(c ^ b));
	TEST_ASSERT(svm::none_of(a ^ b));
	TEST_ASSERT(svm::none_of(c ^ d));

	if constexpr (mask_t::size() > 1)
	{
		std::array<bool, mask_t::size()> a_vals;
		for (std::size_t i = 0; i < mask_t::size(); ++i) a_vals[i] = i % 2;
		std::array<bool, mask_t::size()> b_vals;
		for (std::size_t i = 0; i < mask_t::size(); ++i) b_vals[i] = false;
		std::array<bool, mask_t::size()> c_vals;
		for (std::size_t i = 0; i < mask_t::size(); ++i) c_vals[i] = true;

		a.copy_from(a_vals.data(), svm::element_aligned);
		b.copy_from(b_vals.data(), svm::element_aligned);
		c.copy_from(c_vals.data(), svm::element_aligned);

		TEST_ASSERT(svm::some_of(a));
		TEST_ASSERT(!svm::some_of(b));
		TEST_ASSERT(!svm::some_of(c));

		TEST_ASSERT(svm::popcount(a) == mask_t::size() / 2);
		TEST_ASSERT(svm::popcount(b) == 0);
		TEST_ASSERT(svm::popcount(c) == mask_t::size());

		TEST_ASSERT(svm::find_first_set(a) == 1);
		TEST_ASSERT(svm::find_first_set(c) == 0);
		TEST_ASSERT(svm::find_last_set(a) == mask_t::size() - 1);
		TEST_ASSERT(svm::find_last_set(c) == mask_t::size() - 1);

		std::array<bool, mask_t::size()> tmp = {};

		a.copy_to(tmp.data(), svm::element_aligned);
		TEST_ASSERT(tmp == a_vals);

		b.copy_to(tmp.data(), svm::element_aligned);
		TEST_ASSERT(tmp == b_vals);

		c.copy_to(tmp.data(), svm::element_aligned);
		TEST_ASSERT(tmp == c_vals);
	}
}

int main()
{
	test_mask<float, svm::simd_abi::scalar>();
	test_mask<float, svm::simd_abi::fixed_size<4>>();
	test_mask<float, svm::simd_abi::fixed_size<8>>();
	test_mask<float, svm::simd_abi::fixed_size<16>>();
	test_mask<float, svm::simd_abi::fixed_size<32>>();
	test_mask<float, svm::simd_abi::ext::aligned_vector<4, 16>>();
	test_mask<float, svm::simd_abi::ext::aligned_vector<8, 16>>();
	test_mask<float, svm::simd_abi::ext::aligned_vector<16, 16>>();
	test_mask<float, svm::simd_abi::ext::aligned_vector<32, 16>>();

	svm::simd<float, svm::simd_abi::fixed_size<4>> a = {1.0f}, b = {-1.0f}, d = {0.0f};
	const auto c = -a;

	TEST_ASSERT(svm::all_of(a != b));
	TEST_ASSERT(svm::none_of(a == b));
	TEST_ASSERT(svm::all_of(!(a == b)));

	TEST_ASSERT(svm::all_of(c == b));
	TEST_ASSERT(svm::none_of(c != b));
	TEST_ASSERT(svm::none_of(!(c == b)));

	TEST_ASSERT(svm::all_of(a + b == d));
	TEST_ASSERT(svm::all_of(d - a == b));
	TEST_ASSERT(svm::all_of(d - b == a));
	TEST_ASSERT(svm::all_of(b + b == b * decltype(b){2.0f}));

	TEST_ASSERT(svm::reduce(d) == 0.0f);
	TEST_ASSERT(svm::reduce(a) == 1.0f * a.size());
	TEST_ASSERT(svm::reduce(b) == -1.0f * a.size());

	const auto ia = svm::static_simd_cast<std::int32_t>(a);
	const auto ib = svm::static_simd_cast<std::int32_t>(b);
	const auto ic = svm::static_simd_cast<std::int32_t>(c);
	const auto id = svm::static_simd_cast<std::int32_t>(d);

	TEST_ASSERT(svm::all_of(svm::static_simd_cast<float>(ia) == a));
	TEST_ASSERT(svm::all_of(svm::static_simd_cast<float>(ib) == b));
	TEST_ASSERT(svm::all_of(svm::static_simd_cast<float>(ic) == c));
	TEST_ASSERT(svm::all_of(svm::static_simd_cast<float>(id) == d));

	TEST_ASSERT(svm::all_of(ia != ib));
	TEST_ASSERT(svm::none_of(ia == ib));
	TEST_ASSERT(svm::all_of(!(ia == ib)));

	TEST_ASSERT(svm::all_of(ic == ib));
	TEST_ASSERT(svm::none_of(ic != ib));
	TEST_ASSERT(svm::none_of(!(ic == ib)));

	TEST_ASSERT(svm::all_of(ia + ib == id));
	TEST_ASSERT(svm::all_of(id - ia == ib));
	TEST_ASSERT(svm::all_of(id - ib == ia));
	TEST_ASSERT(svm::all_of(ib + ib == ib * decltype(ib){2}));

	const auto b2 = svm::split_by<2>(b);
	const auto c2 = svm::split_by<2>(c);

	TEST_ASSERT(svm::all_of(b2[0] == b2[0]));
	TEST_ASSERT(svm::all_of(b2[1] == b2[1]));
	TEST_ASSERT(svm::all_of(b2[0] == c2[0]));
	TEST_ASSERT(svm::all_of(b2[1] == c2[1]));

	TEST_ASSERT(svm::all_of(decltype(b){svm::concat(b2)} == b));
	TEST_ASSERT(svm::all_of(decltype(b){svm::concat(b2)} == c));
	TEST_ASSERT(svm::all_of(decltype(b){svm::concat(c2)} == c));

	const auto b4 = svm::split_by<4>(b);
	const auto c4 = svm::split_by<4>(c);

	TEST_ASSERT(svm::all_of(b4[0] == b4[0]));
	TEST_ASSERT(svm::all_of(b4[1] == b4[1]));
	TEST_ASSERT(svm::all_of(b4[2] == b4[2]));
	TEST_ASSERT(svm::all_of(b4[3] == b4[3]));
	TEST_ASSERT(svm::all_of(b4[0] == c4[0]));
	TEST_ASSERT(svm::all_of(b4[1] == c4[1]));
	TEST_ASSERT(svm::all_of(b4[2] == c4[2]));
	TEST_ASSERT(svm::all_of(b4[3] == c4[3]));

	TEST_ASSERT(svm::all_of(decltype(b){svm::concat(b4)} == b));
	TEST_ASSERT(svm::all_of(decltype(b){svm::concat(b4)} == c));
	TEST_ASSERT(svm::all_of(decltype(b){svm::concat(c4)} == c));

	alignas(decltype(d)) std::array<float, 4> d_vals;
	for (std::size_t i = 0; i < d_vals.size(); ++i) d_vals[i] = i % 2 ? b[i] : a[i];
	alignas(decltype(d)::mask_type) std::array<bool, 4> mask_vals;
	for (std::size_t i = 0; i < mask_vals.size(); ++i) mask_vals[i] = i % 2;

	typename decltype(d)::mask_type mask;
	d.copy_from(d_vals.data(), svm::vector_aligned);
	mask.copy_from(mask_vals.data(), svm::vector_aligned);

	TEST_ASSERT(svm::all_of(svm::ext::blend(a, where(mask, b)) == d));

	alignas(decltype(d)) std::array<float, 4> v_tmp;
	where(mask, b).copy_to(v_tmp.data(), svm::vector_aligned);
	where(!mask, a).copy_to(v_tmp.data(), svm::vector_aligned);

	TEST_ASSERT(v_tmp == d_vals);
}
