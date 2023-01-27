/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "../type_fwd.hpp"

namespace dpm::detail
{
	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE T setzero() noexcept requires (std::same_as<T, m64x4> || std::same_as<T, f64x4>)
	{
		return std::bit_cast<T>(_mm256_setzero_pd());
	}
	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE T setones() noexcept requires (std::same_as<T, m64x4> || std::same_as<T, f64x4>)
	{
		const auto tmp = _mm256_undefined_pd();
		return std::bit_cast<T>(_mm256_cmp_pd(tmp, tmp, _MM_CMPINT_EQ));
	}

	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE T setzero() noexcept requires (std::same_as<T, m32x8> || std::same_as<T, f32x8>)
	{
		return std::bit_cast<T>(_mm256_setzero_ps());
	}
	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE T setones() noexcept requires (std::same_as<T, m32x8> || std::same_as<T, f32x8>)
	{
		const auto tmp = _mm256_undefined_ps();
		return std::bit_cast<T>(_mm256_cmp_ps(tmp, tmp, _MM_CMPINT_EQ));
	}

	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE T setzero() noexcept
	requires (std::same_as<T, i64x4> || std::same_as<T, u64x4> ||
	          std::same_as<T, i32x8> || std::same_as<T, u32x8> ||
	          std::same_as<T, i16x16> || std::same_as<T, u16x16> ||
	          std::same_as<T, i8x32> || std::same_as<T, u8x32>)
	{
		return std::bit_cast<T>(_mm256_setzero_si256());
	}
	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE T setones() noexcept
	requires (std::same_as<T, i64x4> || std::same_as<T, u64x4> ||
	          std::same_as<T, i32x8> || std::same_as<T, u32x8> ||
	          std::same_as<T, i16x16> || std::same_as<T, u16x16> ||
	          std::same_as<T, i8x32> || std::same_as<T, u8x32>)
	{
		const auto tmp = _mm256_undefined_si256();
		return std::bit_cast<T>(_mm256_cmpeq_epi64(tmp, tmp));
	}
}