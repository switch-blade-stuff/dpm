/*
 * Created by switchblade on 2023-01-01.
 */

#include <algorithm>
#include <array>

#include <emmintrin.h>
#include <immintrin.h>

#include <svm/detail/x86/cpuid.hpp>
#include <svm/detail/dispatch.hpp>

union alignas(__m256) data_t
{
	std::array<float, 4> elems;
	__m256 simd;
};

static data_t SVM_TARGET("avx") add_generic(data_t a, data_t b) noexcept
{
	data_t out;
	out.elems[0] = a.elems[0] + b.elems[0];
	out.elems[1] = a.elems[1] + b.elems[1];
	out.elems[2] = a.elems[2] + b.elems[2];
	out.elems[3] = a.elems[3] + b.elems[3];
	return out;
}

static data_t SVM_TARGET("avx") add_sse(data_t a, data_t b) noexcept
{
	data_t out;
	out.simd = _mm256_castps128_ps256(_mm_add_ps(_mm256_castps256_ps128(a.simd), _mm256_castps256_ps128(b.simd)));
	return out;
}
static data_t SVM_TARGET("avx") add(data_t a, data_t b) noexcept
{
	static constinit svm::detail::dispatcher dispatch_add = []()
	{
		if (svm::detail::cpuid::has_sse())
			return add_sse;
		else
			return add_generic;
	};
	return dispatch_add(a, b);
}

int main()
{
	data_t a, b, c;
	std::ranges::fill(a.elems, 1.0f);
	std::ranges::fill(b.elems, 2.0f);
	std::ranges::fill(c.elems, 3.0f);

	const auto d = add(a, b);
	printf("a + b == c: %i\n", c.elems == d.elems);
}
