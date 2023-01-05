/*
 * Created by switchblade on 2023-01-01.
 */

#include <algorithm>
#include <array>

#include <emmintrin.h>
#include <immintrin.h>

#define SVM_DEBUG

#include <svm/type.hpp>

#define TEST_ASSERT(x) SVM_ASSERT((x), nullptr)

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

	TEST_ASSERT(svm::all_of(!c == a));
	TEST_ASSERT(svm::all_of(!a == c));

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
	test_mask<float, svm::simd_abi::aligned_vector<4, 16>>();
	test_mask<float, svm::simd_abi::aligned_vector<8, 16>>();
	test_mask<float, svm::simd_abi::aligned_vector<16, 16>>();
	test_mask<float, svm::simd_abi::aligned_vector<32, 16>>();

	svm::simd<float, svm::simd_abi::sse<float>> a = {1.0f}, b = {-1.0f}, d = {0.0f};
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
}
