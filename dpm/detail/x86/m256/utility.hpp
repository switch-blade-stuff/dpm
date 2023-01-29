/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "../type_fwd.hpp"

namespace dpm::detail
{
	template<std::same_as<__m256> V, typename F, typename... Args>
	[[nodiscard]] DPM_FORCEINLINE V mux_128(Args &&...args, F op)
	{
		const auto l = op(_mm256_extractf128_ps(std::bit_cast<__m256>(args), 0)...);
		const auto h = op(_mm256_extractf128_ps(std::bit_cast<__m256>(args), 1)...);
		return _mm256_set_m128(h, l);
	}
	template<std::same_as<__m256d> V, typename F, typename... Args>
	[[nodiscard]] DPM_FORCEINLINE V mux_128(Args &&...args, F op)
	{
		const auto l = op(_mm256_extractf128_pd(std::bit_cast<__m256d>(args), 0)...);
		const auto h = op(_mm256_extractf128_pd(std::bit_cast<__m256d>(args), 1)...);
		return _mm256_set_m128d(h, l);
	}
	template<std::same_as<__m256i> V, typename F, typename... Args>
	[[nodiscard]] DPM_FORCEINLINE V mux_128(Args &&...args, F op)
	{
		const auto l = op(_mm256_extractf128_si256(std::bit_cast<__m256i>(args), 0)...);
		const auto h = op(_mm256_extractf128_si256(std::bit_cast<__m256i>(args), 1)...);
		return _mm256_set_m128i(h, l);
	}
}