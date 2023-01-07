/*
 * Created by switchblade on 2023-01-06.
 */

#include <svm/simd.hpp>

constexpr static std::size_t N = 3;

void test(svm::simd<float, svm::simd_abi::fixed_size<N>> value, std::array<float, N> &mem)
{
	value.copy_to(mem.data(), svm::overaligned<4>);
}
void test(svm::simd<float, svm::simd_abi::fixed_size<N>> value, svm::simd_mask<float, svm::simd_abi::fixed_size<N>> mask, std::array<float, N> &mem)
{
	where(mask, value).copy_to(mem.data(), svm::overaligned<16>);
}