/*
 * Created by switchblade on 2023-01-07.
 */

#include <svm/simd.hpp>

constexpr static std::size_t N = 4;

auto all_of(svm::simd_mask<float, svm::simd_abi::fixed_size<N>> value)
{
	return svm::any_of(value);
}

auto popcnt(svm::simd_mask<float, svm::simd_abi::fixed_size<N>> value)
{
	return svm::popcount(value);
}

auto find_first(svm::simd_mask<float, svm::simd_abi::fixed_size<N>> value)
{
	return svm::find_first_set(value);
}
auto find_last(svm::simd_mask<float, svm::simd_abi::fixed_size<N>> value)
{
	return svm::find_last_set(value);
}
