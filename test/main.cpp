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
		a[0] = false;
		TEST_ASSERT(svm::some_of(a));
		TEST_ASSERT(!svm::some_of(b));
		TEST_ASSERT(!svm::some_of(c));
	}
}

int main()
{
	test_mask<int, svm::simd_abi::scalar>();
	test_mask<int, svm::simd_abi::fixed_size<4>>();
	test_mask<int, svm::simd_abi::aligned_vector<4, 16>>();
}
