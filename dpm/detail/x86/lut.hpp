/*
 * Created by switchblade on 2023-02-18.
 */

#pragma once

#include "../../debug.hpp"
#include "utility.hpp"

namespace dpm::detail
{
	template<typename I, std::size_t N, typename V, std::size_t... Is>
	[[nodiscard]] inline const V &assert_lut_idx(std::index_sequence<Is...>, const V &v_idx, [[maybe_unused]] DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		[[maybe_unused]] const auto idx = reinterpret_cast<const alias_t<I> *>(&v_idx);
		/* Assert with source location to properly report the offending function name (rather than asssert_lut_index). */
		DPM_ASSERT_MSG_LOC(((static_cast<std::size_t>(idx[Is]) < N) && ...), "lut index out of bounds", loc);
		return v_idx;
	}
	template<typename I, std::size_t N, typename V>
	[[nodiscard]] inline const V &assert_lut_idx(const V &v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return assert_lut_idx<I, N>(std::make_index_sequence<sizeof(V) / sizeof(I)>{}, v_idx, loc);
	}

#ifdef DPM_HAS_AVX2
	template<std::same_as<__m128> V, integral_of_size<4> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm_i32gather_ps(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
	template<std::same_as<__m256> V, integral_of_size<4> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m256i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm256_i32gather_ps(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
	template<std::same_as<__m128> V, integral_of_size<8> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm_i64gather_ps(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
	template<std::same_as<__m128> V, integral_of_size<8> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m256i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm256_i64gather_ps(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}

	template<std::same_as<__m128d> V, integral_of_size<4> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm_i32gather_pd(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
	template<std::same_as<__m256d> V, integral_of_size<4> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm256_i32gather_pd(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
	template<std::same_as<__m128d> V, integral_of_size<8> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm_i64gather_pd(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
	template<std::same_as<__m256d> V, integral_of_size<8> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m256i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm256_i64gather_pd(data.data(), (assert_lut_idx<I, N>(v_idx, loc)), Scale);
	}
#else
	template<std::same_as<__m128> V, integral_of_size<4> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm_set_ps(
				data[static_cast<std::size_t>(idx[3] * Scale / 4)],
				data[static_cast<std::size_t>(idx[2] * Scale / 4)],
				data[static_cast<std::size_t>(idx[1] * Scale / 4)],
				data[static_cast<std::size_t>(idx[0] * Scale / 4)]
		);
	}
	template<std::same_as<__m128> V, integral_of_size<8> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm_set_ps(
				0.0f, 0.0f,
				data[static_cast<std::size_t>(idx[1] * Scale / 4)],
				data[static_cast<std::size_t>(idx[0] * Scale / 4)]
		);
	}
	template<std::same_as<__m128d> V, integral_of_size<4> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm_set_pd(
				data[static_cast<std::size_t>(idx[1] * Scale / 8)],
				data[static_cast<std::size_t>(idx[0] * Scale / 8)]
		);
	}
	template<std::same_as<__m128d> V, integral_of_size<8> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm_set_pd(
				data[static_cast<std::size_t>(idx[1] * Scale / 8)],
				data[static_cast<std::size_t>(idx[0] * Scale / 8)]
		);
	}

#ifdef DPM_HAS_AVX
	template<std::same_as<__m256> V, integral_of_size<4> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m256i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm256_set_ps(
				data[static_cast<std::size_t>(idx[7] * Scale / 4)],
				data[static_cast<std::size_t>(idx[6] * Scale / 4)],
				data[static_cast<std::size_t>(idx[5] * Scale / 4)],
				data[static_cast<std::size_t>(idx[4] * Scale / 4)],
				data[static_cast<std::size_t>(idx[3] * Scale / 4)],
				data[static_cast<std::size_t>(idx[2] * Scale / 4)],
				data[static_cast<std::size_t>(idx[1] * Scale / 4)],
				data[static_cast<std::size_t>(idx[0] * Scale / 4)]
		);
	}
	template<std::same_as<__m128> V, integral_of_size<8> I, int Scale = 4, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const float, N> data, __m256i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm_set_ps(
				data[static_cast<std::size_t>(idx[3] * Scale / 4)],
				data[static_cast<std::size_t>(idx[2] * Scale / 4)],
				data[static_cast<std::size_t>(idx[1] * Scale / 4)],
				data[static_cast<std::size_t>(idx[0] * Scale / 4)]
		);
	}
	template<std::same_as<__m256d> V, integral_of_size<4> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m128i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm256_set_pd(
				data[static_cast<std::size_t>(idx[3] * Scale / 8)],
				data[static_cast<std::size_t>(idx[2] * Scale / 8)],
				data[static_cast<std::size_t>(idx[1] * Scale / 8)],
				data[static_cast<std::size_t>(idx[0] * Scale / 8)]
		);
	}
	template<std::same_as<__m256d> V, integral_of_size<8> I, int Scale = 8, std::size_t N>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(std::span<const double, N> data, __m256i v_idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto idx = reinterpret_cast<const alias_t<I> *>(&assert_lut_idx<I, N>(v_idx, loc));
		return _mm256_set_pd(
				data[static_cast<std::size_t>(idx[3] * Scale / 8)],
				data[static_cast<std::size_t>(idx[2] * Scale / 8)],
				data[static_cast<std::size_t>(idx[1] * Scale / 8)],
				data[static_cast<std::size_t>(idx[0] * Scale / 8)]
		);
	}
#endif
#endif

#ifdef DPM_HAS_AVX
	template<std::same_as<__m256> V, integral_of_size<8> Idx, int Scale = 1>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(const float *data, __m256i idx, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		return _mm256_castps128_ps256(lut_load<__m128, Idx, Scale>(data, idx, loc));
	}
	template<std::same_as<__m256> V, integral_of_size<8> Idx, int Scale = 1>
	[[nodiscard]] DPM_FORCEINLINE V lut_load(const float *data, __m256i idxl, __m256i idxh, DPM_ASSERT_LOC_TYPE loc) noexcept
	{
		const auto vl = lut_load<__m128, Idx, Scale>(data, idxl, loc);
		const auto vh = lut_load<__m128, Idx, Scale>(data, idxh, loc);
		return _mm256_set_m128(vh, vl);
	}
#endif
}