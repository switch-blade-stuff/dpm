/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "traits.hpp"

namespace svm
{
	struct vector_aligned_tag {};
	struct element_aligned_tag {};

	inline constexpr auto vector_aligned = vector_aligned_tag{};
	inline constexpr auto element_aligned = element_aligned_tag{};

	template<>
	struct is_simd_flag_type<vector_aligned_tag> : std::true_type {};
	template<>
	struct is_simd_flag_type<element_aligned_tag> : std::true_type {};

	template<std::size_t>
	struct overaligned_tag {};

	template<std::size_t N>
	inline constexpr auto overaligned = overaligned_tag<N>{};

	template<std::size_t N>
	struct is_simd_flag_type<overaligned_tag<N>> : std::true_type {};
}