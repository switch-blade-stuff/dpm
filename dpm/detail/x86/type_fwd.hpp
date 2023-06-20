/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include <bit>

#include "../generic/type.hpp"
#include "../../utility.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "abi.hpp"

namespace dpm::detail
{
	using simd_abi::detail::x86_overload_128;
	using simd_abi::detail::x86_simd_abi_128;

#ifdef DPM_HAS_AVX
	using simd_abi::detail::x86_overload_256;
	using simd_abi::detail::x86_simd_abi_256;
#endif

	using simd_abi::detail::x86_overload_any;
	using simd_abi::detail::x86_simd_abi_any;

	template<typename T, std::size_t N, std::size_t A>
	using x86_mask = simd_mask<T, avec<N, A>>;
	template<typename T, std::size_t N, std::size_t A>
	using x86_simd = simd<T, avec<N, A>>;

	template<typename, std::size_t>
	struct select_vector;
	template<typename T, std::size_t N>
	using select_vector_t = typename select_vector<T, N>::type;

	template<typename>
	struct movemask_bits : std::integral_constant<std::size_t, 1> {};
	template<typename T> requires(sizeof(T) == 2)
	struct movemask_bits<T> : std::integral_constant<std::size_t, 2> {};
	template<typename T>
	inline constexpr auto movemask_bits_v = movemask_bits<T>::value;

	/* TODO: use custom vector types under GCC to enable indexing without reinterpret_cast. */
	template<>
	struct select_vector<float, 16> { using type = __m128; };

#ifdef DPM_HAS_SSE2
	template<std::integral T>
	struct select_vector<T, 16> { using type = __m128i; };
	template<>
	struct select_vector<double, 16> { using type = __m128d; };
#endif

#ifdef DPM_HAS_AVX
	template<std::integral T>
	struct select_vector<T, 32> { using type = __m256i; };
	template<>
	struct select_vector<float, 32> { using type = __m256; };
	template<>
	struct select_vector<double, 32> { using type = __m256d; };
#endif

#ifdef _MSC_VER
	template<std::ranges::range R>
	static constexpr decltype(auto) native_bit_cast(__m128 value) noexcept
	{
		R result = {};
		std::copy_n(value.m128_f32, 4, std::ranges::begin(result));
		return result;
	}
	template<std::same_as<__m128> V>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept
	{
		__m128 result = {.m128_f32 = {}};
		std::copy_n(std::ranges::begin(value), 4, result.m128_f32);
		return result;
	}

	template<std::same_as<float>>
	static constexpr decltype(auto) native_index_at(__m128 &value, std::size_t i) noexcept { return value.m128_f32[i]; }
	template<std::same_as<float>>
	static constexpr decltype(auto) native_index_at(const __m128 &value, std::size_t i) noexcept { return value.m128_f32[i]; }

#ifdef DPM_HAS_SSE2
	template<std::ranges::range R>
	static constexpr decltype(auto) native_bit_cast(__m128d value) noexcept
	{
		R result = {};
		std::copy_n(value.m128d_f64, 2, std::ranges::begin(result));
		return result;
	}
	template<std::same_as<__m128d> V>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept
	{
		__m128d result = {.m128d_f64 = {}};
		std::copy_n(std::ranges::begin(value), 2, result.m128d_f64);
		return result;
	}

	template<std::same_as<double>>
	static constexpr decltype(auto) native_index_at(__m128d &value, std::size_t i) noexcept { return value.m128d_f64[i]; }
	template<std::same_as<double>>
	static constexpr decltype(auto) native_index_at(const __m128d &value, std::size_t i) noexcept { return value.m128d_f64[i]; }

	template<std::ranges::range R>
	static constexpr decltype(auto) native_bit_cast(__m128i value) noexcept
	{
		std::array<std::uint64_t, 2> result = {};
		std::copy_n(value.m128i_u64, 2, std::ranges::begin(result));
		return std::bit_cast<R>(result);
	}
	template<std::same_as<__m128i> V>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept
	{
		__m128i result = {.m128i_u64 = {}};
		const auto tmp = std::bit_cast<std::array<std::uint64_t, 2>>(value);
		std::copy_n(std::ranges::begin(tmp), 2, result.m128i_u64);
		return result;
	}

	template<signed_integral_of_size<1>>
	static constexpr decltype(auto) native_index_at(__m128i &value, std::size_t i) noexcept { return value.m128i_i8[i]; }
	template<signed_integral_of_size<1>>
	static constexpr decltype(auto) native_index_at(const __m128i &value, std::size_t i) noexcept { return value.m128i_i8[i]; }

	template<unsigned_integral_of_size<1>>
	static constexpr decltype(auto) native_index_at(__m128i &value, std::size_t i) noexcept { return value.m128i_u8[i]; }
	template<unsigned_integral_of_size<1>>
	static constexpr decltype(auto) native_index_at(const __m128i &value, std::size_t i) noexcept { return value.m128i_u8[i]; }
#endif
#ifdef DPM_HAS_AVX
	template<std::ranges::range R>
	static constexpr decltype(auto) native_bit_cast(__m256 value) noexcept
	{
		R result = {};
		std::copy_n(value.m256_f32, 4, std::ranges::begin(result));
		return result;
	}
	template<std::same_as<__m256> V>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept
	{
		__m256 result = {};
		std::copy_n(std::ranges::begin(value), 4, result.m256_f32);
		return result;
	}

	template<std::ranges::range R>
	static constexpr decltype(auto) native_bit_cast(__m256d value) noexcept
	{
		R result = {};
		std::copy_n(value.m256d_f64, 2, std::ranges::begin(result));
		return result;
	}
	template<std::same_as<__m256d> V>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept
	{
		__m256d result = {};
		std::copy_n(std::ranges::begin(value), 2, result.m256d_f64);
		return result;
	}

	template<std::ranges::range R>
	static constexpr decltype(auto) native_bit_cast(__m256i value) noexcept
	{
		std::array<std::uint64_t, 4> result = {};
		std::copy_n(value.m256i_u64, 4, std::ranges::begin(result));
		return std::bit_cast<R>(result);
	}
	template<std::same_as<__m256i> V>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept
	{
		__m256i result = {};
		const auto tmp = std::bit_cast<std::array<std::uint64_t, 4>>(value);
		std::copy_n(std::ranges::begin(tmp), 4, result.m256i_u64);
		return result;
	}
#endif
#elif defined(__GNUC__) && !defined(__clang__) /* CLang does not support constexpr bit_cast or element access with vector types yet. */
	template<typename T>
	static constexpr decltype(auto) native_bit_cast(auto &&value) noexcept { return std::bit_cast<T>(value); }
	template<typename T>
	static constexpr decltype(auto) native_index_at(auto &&value, std::size_t i) noexcept { return value[i]; }
#else
	template<typename T>
	static inline decltype(auto) native_bit_cast(auto &&value) noexcept { return std::bit_cast<T>(value); }
	template<typename T>
	static inline decltype(auto) native_index_at(auto &&value, std::size_t i) noexcept { return value[i]; }
#endif
}

#endif