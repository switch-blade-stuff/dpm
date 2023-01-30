/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "type_fwd.hpp"
#include "cvt.hpp"

namespace dpm::detail
{
	template<typename T, typename U, typename V>
	DPM_FORCEINLINE void cast_copy(U *dst, const V &src) noexcept requires (std::same_as<T, U> || (std::integral<T> && std::integral<U> && sizeof(T) == sizeof(U)))
	{
		*reinterpret_cast<V *>(dst) = src;
	}
	template<typename T, typename U, typename V>
	DPM_FORCEINLINE void cast_copy(V &dst, const U *src) noexcept requires (std::same_as<T, U> || (std::integral<T> && std::integral<U> && sizeof(T) == sizeof(U)))
	{
		dst = *reinterpret_cast<const V *>(src);
	}

	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setzero() noexcept requires (sizeof(V) == 16)
	{
		return std::bit_cast<V>(_mm_setzero_ps());
	}
	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setones() noexcept requires (sizeof(V) == 16)
	{
#ifndef DPM_HAS_SSE2
		const auto tmp = _mm_setzero_ps();
		return std::bit_cast<V>(_mm_cmpeq_ps(tmp, tmp));
#else
		const auto tmp = _mm_undefined_si128();
		return std::bit_cast<V>(_mm_cmpeq_epi32(tmp, tmp));
#endif
	}

	template<std::same_as<__m128> V>
	[[nodiscard]] DPM_FORCEINLINE V set(const float (&vals)[4]) noexcept { return _mm_set_ps(vals[3], vals[2], vals[1], vals[0]); }
	template<std::same_as<__m128> V>
	[[nodiscard]] DPM_FORCEINLINE V fill(float value) noexcept { return _mm_set1_ps(value); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_ps(std::bit_cast<__m128>(x)));
	}

#ifdef DPM_HAS_SSE2
	template<std::same_as<__m128d> V>
	[[nodiscard]] DPM_FORCEINLINE V set(const double (&vals)[2]) noexcept { return _mm_set_pd(vals[1], vals[0]); }
	template<std::same_as<__m128i> V, integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[16]) noexcept
	{
		return _mm_set_epi8(
				static_cast<std::int8_t>(vals[15]), static_cast<std::int8_t>(vals[14]),
				static_cast<std::int8_t>(vals[13]), static_cast<std::int8_t>(vals[12]),
				static_cast<std::int8_t>(vals[11]), static_cast<std::int8_t>(vals[10]),
				static_cast<std::int8_t>(vals[9]), static_cast<std::int8_t>(vals[8]),
				static_cast<std::int8_t>(vals[7]), static_cast<std::int8_t>(vals[6]),
				static_cast<std::int8_t>(vals[5]), static_cast<std::int8_t>(vals[4]),
				static_cast<std::int8_t>(vals[3]), static_cast<std::int8_t>(vals[2]),
				static_cast<std::int8_t>(vals[1]), static_cast<std::int8_t>(vals[0])
		);
	}
	template<std::same_as<__m128i> V, integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[8]) noexcept
	{
		return _mm_set_epi16(
				static_cast<std::int16_t>(vals[7]), static_cast<std::int16_t>(vals[6]),
				static_cast<std::int16_t>(vals[5]), static_cast<std::int16_t>(vals[4]),
				static_cast<std::int16_t>(vals[3]), static_cast<std::int16_t>(vals[2]),
				static_cast<std::int16_t>(vals[1]), static_cast<std::int16_t>(vals[0])
		);
	}
	template<std::same_as<__m128i> V, integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[4]) noexcept
	{
		return _mm_set_epi32(
				static_cast<std::int32_t>(vals[3]),
				static_cast<std::int32_t>(vals[2]),
				static_cast<std::int32_t>(vals[1]),
				static_cast<std::int32_t>(vals[0])
		);
	}
	template<std::same_as<__m128i> V, integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[2]) noexcept
	{
		return _mm_set_epi64x(
				static_cast<std::int64_t>(vals[1]),
				static_cast<std::int64_t>(vals[0])
		);
	}

	template<std::same_as<__m128d> V>
	[[nodiscard]] DPM_FORCEINLINE V fill(double value) noexcept { return _mm_set1_pd(value); }
	template<std::same_as<__m128i> V, integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi8(static_cast<std::int8_t>(value)); }
	template<std::same_as<__m128i> V, integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi16(static_cast<std::int16_t>(value)); }
	template<std::same_as<__m128i> V, integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi32(static_cast<std::int32_t>(value)); }
	template<std::same_as<__m128i> V, integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi64x(static_cast<std::int64_t>(value)); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_epi8(std::bit_cast<__m128i>(x)));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_pd(std::bit_cast<__m128>(x)));
	}

	template<std::same_as<float> T, signed_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128 &src) noexcept { *reinterpret_cast<__m128i *>(dst) = _mm_cvtps_epi32(src); }
	template<signed_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128i &src) noexcept { *reinterpret_cast<__m128 *>(dst) = _mm_cvtepi32_ps(src); }
	template<std::same_as<float> T, unsigned_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128 &src) noexcept { *reinterpret_cast<__m128i *>(dst) = cvt_f32_u32(src); }
	template<unsigned_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128i &src) noexcept { *reinterpret_cast<__m128 *>(dst) = cvt_u32_f32(src); }

	template<std::same_as<float> T, signed_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(__m128 &dst, const U *src) noexcept { dst = _mm_cvtepi32_ps(*reinterpret_cast<const __m128i *>(src)); }
	template<signed_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(__m128i &dst, const U *src) noexcept { dst = _mm_cvtps_epi32(*reinterpret_cast<const __m128 *>(src)); }
	template<std::same_as<float> T, unsigned_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(__m128 &dst, const U *src) noexcept { dst = cvt_u32_f32(*reinterpret_cast<const __m128i *>(src)); }
	template<unsigned_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(__m128i &dst, const U *src) noexcept { dst = cvt_f32_u32(*reinterpret_cast<const __m128 *>(src)); }

	template<std::same_as<double> T, signed_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128d &src) noexcept { *reinterpret_cast<__m128i *>(dst) = cvt_f64_i64(src); }
	template<signed_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128i &src) noexcept { *reinterpret_cast<__m128 *>(dst) = cvt_i64_f64(src); }
	template<std::same_as<double> T, unsigned_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128d &src) noexcept { *reinterpret_cast<__m128i *>(dst) = cvt_f64_u64(src); }
	template<unsigned_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m128i &src) noexcept { *reinterpret_cast<__m128 *>(dst) = cvt_u64_f64(src); }

	template<std::same_as<double> T, signed_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(__m128d &dst, const U *src) noexcept { dst = cvt_i64_f64(*reinterpret_cast<const __m128i *>(src)); }
	template<signed_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(__m128i &dst, const U *src) noexcept { dst = cvt_f64_i64(*reinterpret_cast<const __m128d *>(src)); }
	template<std::same_as<double> T, unsigned_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(__m128d &dst, const U *src) noexcept { dst = cvt_u64_f64(*reinterpret_cast<const __m128i *>(src)); }
	template<unsigned_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(__m128i &dst, const U *src) noexcept { dst = cvt_f64_u64(*reinterpret_cast<const __m128d *>(src)); }
#endif

#ifdef DPM_HAS_AVX
	/* In case 256-bit operations are not available, they can be faked via 2 128-bit operations. */
	template<std::same_as<__m256> V, typename F, typename... Args>
	[[nodiscard]] DPM_FORCEINLINE V mux_128x2(F op, Args &&...args)
	{
		const auto l = op(_mm256_extractf128_ps(std::bit_cast<__m256>(args), 0)...);
		const auto h = op(_mm256_extractf128_ps(std::bit_cast<__m256>(args), 1)...);
		return _mm256_set_m128(h, l);
	}
	template<std::same_as<__m256d> V, typename F, typename... Args>
	[[nodiscard]] DPM_FORCEINLINE V mux_128x2(F op, Args &&...args)
	{
		const auto l = op(_mm256_extractf128_pd(std::bit_cast<__m256d>(args), 0)...);
		const auto h = op(_mm256_extractf128_pd(std::bit_cast<__m256d>(args), 1)...);
		return _mm256_set_m128d(h, l);
	}
	template<std::same_as<__m256i> V, typename F, typename... Args>
	[[nodiscard]] DPM_FORCEINLINE V mux_128x2(F op, Args &&...args)
	{
		const auto l = op(_mm256_extractf128_si256(std::bit_cast<__m256i>(args), 0)...);
		const auto h = op(_mm256_extractf128_si256(std::bit_cast<__m256i>(args), 1)...);
		return _mm256_set_m128i(h, l);
	}

	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setzero() noexcept requires (sizeof(V) == 32)
	{
		return std::bit_cast<V>(_mm256_setzero_ps());
	}
	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setones() noexcept requires (sizeof(V) == 32)
	{
#ifndef DPM_HAS_AVX2
		const auto tmp = _mm256_setzero_ps();
		return std::bit_cast<V>(_mm256_cmp_ps(tmp, tmp, _CMP_EQ_OS));
#else
		const auto tmp = _mm256_undefined_si256();
		return std::bit_cast<V>(_mm256_cmpeq_epi32(tmp, tmp));
#endif
	}

	template<std::same_as<__m256> V>
	[[nodiscard]] DPM_FORCEINLINE V set(const float (&vals)[8]) noexcept { return _mm256_set_ps(vals[7], vals[6], vals[5], vals[4], vals[3], vals[2], vals[1], vals[0]); }
	template<std::same_as<__m256d> V>
	[[nodiscard]] DPM_FORCEINLINE V set(const double (&vals)[4]) noexcept { return _mm256_set_pd(vals[3], vals[2], vals[1], vals[0]); }
	template<std::same_as<__m256i> V, integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[32]) noexcept
	{
		return _mm256_set_epi8(
				static_cast<std::int8_t>(vals[31]), static_cast<std::int8_t>(vals[30]),
				static_cast<std::int8_t>(vals[29]), static_cast<std::int8_t>(vals[28]),
				static_cast<std::int8_t>(vals[27]), static_cast<std::int8_t>(vals[26]),
				static_cast<std::int8_t>(vals[25]), static_cast<std::int8_t>(vals[24]),
				static_cast<std::int8_t>(vals[23]), static_cast<std::int8_t>(vals[22]),
				static_cast<std::int8_t>(vals[21]), static_cast<std::int8_t>(vals[20]),
				static_cast<std::int8_t>(vals[19]), static_cast<std::int8_t>(vals[18]),
				static_cast<std::int8_t>(vals[17]), static_cast<std::int8_t>(vals[16]),
				static_cast<std::int8_t>(vals[15]), static_cast<std::int8_t>(vals[14]),
				static_cast<std::int8_t>(vals[13]), static_cast<std::int8_t>(vals[12]),
				static_cast<std::int8_t>(vals[11]), static_cast<std::int8_t>(vals[10]),
				static_cast<std::int8_t>(vals[9]), static_cast<std::int8_t>(vals[8]),
				static_cast<std::int8_t>(vals[7]), static_cast<std::int8_t>(vals[6]),
				static_cast<std::int8_t>(vals[5]), static_cast<std::int8_t>(vals[4]),
				static_cast<std::int8_t>(vals[3]), static_cast<std::int8_t>(vals[2]),
				static_cast<std::int8_t>(vals[1]), static_cast<std::int8_t>(vals[0])
		);
	}
	template<std::same_as<__m256i> V, integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[16]) noexcept
	{
		return _mm256_set_epi16(
				static_cast<std::int16_t>(vals[15]), static_cast<std::int16_t>(vals[14]),
				static_cast<std::int16_t>(vals[13]), static_cast<std::int16_t>(vals[12]),
				static_cast<std::int16_t>(vals[11]), static_cast<std::int16_t>(vals[10]),
				static_cast<std::int16_t>(vals[9]), static_cast<std::int16_t>(vals[8]),
				static_cast<std::int16_t>(vals[7]), static_cast<std::int16_t>(vals[6]),
				static_cast<std::int16_t>(vals[5]), static_cast<std::int16_t>(vals[4]),
				static_cast<std::int16_t>(vals[3]), static_cast<std::int16_t>(vals[2]),
				static_cast<std::int16_t>(vals[1]), static_cast<std::int16_t>(vals[0])
		);
	}
	template<std::same_as<__m256i> V, integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[8]) noexcept
	{
		return _mm256_set_epi32(
				static_cast<std::int32_t>(vals[7]), static_cast<std::int32_t>(vals[6]),
				static_cast<std::int32_t>(vals[5]), static_cast<std::int32_t>(vals[4]),
				static_cast<std::int32_t>(vals[3]), static_cast<std::int32_t>(vals[2]),
				static_cast<std::int32_t>(vals[1]), static_cast<std::int32_t>(vals[0])
		);
	}
	template<std::same_as<__m256i> V, integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE V set(const T (&vals)[4]) noexcept
	{
		return _mm256_set_epi64x(
				static_cast<std::int32_t>(vals[3]),
				static_cast<std::int32_t>(vals[2]),
				static_cast<std::int32_t>(vals[1]),
				static_cast<std::int32_t>(vals[0])
		);
	}

	template<std::same_as<__m256> V>
	[[nodiscard]] DPM_FORCEINLINE V fill(float value) noexcept { return _mm256_set1_ps(value); }
	template<std::same_as<__m256d> V>
	[[nodiscard]] DPM_FORCEINLINE V fill(double value) noexcept { return _mm256_set1_pd(value); }
	template<std::same_as<__m256i> V, integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi8(static_cast<std::int8_t>(value)); }
	template<std::same_as<__m256i> V, integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi16(static_cast<std::int16_t>(value)); }
	template<std::same_as<__m256i> V, integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi32(static_cast<std::int32_t>(value)); }
	template<std::same_as<__m256i> V, integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi64x(static_cast<std::int64_t>(value)); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 32)
	{
#ifdef DPM_HAS_AVX2
		return static_cast<std::size_t>(_mm256_movemask_epi8(std::bit_cast<__m256i>(x)));
#else
		const auto ml = movemask<T>(_mm256_extractf128_si256(std::bit_cast<__m256i>(x), 0));
		const auto mh = movemask<T>(_mm256_extractf128_si256(std::bit_cast<__m256i>(x), 1));
		return (mh << (sizeof(V) / sizeof(T) * movemask_bits_v<T>)) | ml;
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 4 && sizeof(V) == 32)
	{
		return static_cast<std::size_t>(_mm256_movemask_ps(std::bit_cast<__m256>(x)));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 8 && sizeof(V) == 32)
	{
		return static_cast<std::size_t>(_mm256_movemask_pd(std::bit_cast<__m256d>(x)));
	}

	template<std::same_as<float> T, signed_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256 &src) noexcept { *reinterpret_cast<__m256i *>(dst) = _mm256_cvtps_epi32(src); }
	template<signed_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256i &src) noexcept { *reinterpret_cast<__m256 *>(dst) = _mm256_cvtepi32_ps(src); }
	template<std::same_as<float> T, unsigned_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256 &src) noexcept { *reinterpret_cast<__m256i *>(dst) = cvt_f32_u32(src); }
	template<unsigned_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256i &src) noexcept { *reinterpret_cast<__m256 *>(dst) = cvt_u32_f32(src); }

	template<std::same_as<float> T, signed_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(__m256 &dst, const U *src) noexcept { dst = _mm256_cvtepi32_ps(*reinterpret_cast<const __m256i *>(src)); }
	template<signed_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(__m256i &dst, const U *src) noexcept { dst = _mm256_cvtps_epi32(*reinterpret_cast<const __m256 *>(src)); }
	template<std::same_as<float> T, unsigned_integral_of_size<4> U>
	DPM_FORCEINLINE void cast_copy(__m256 &dst, const U *src) noexcept { dst = cvt_u32_f32(*reinterpret_cast<const __m256i *>(src)); }
	template<unsigned_integral_of_size<4> T, std::same_as<float> U>
	DPM_FORCEINLINE void cast_copy(__m256i &dst, const U *src) noexcept { dst = cvt_f32_u32(*reinterpret_cast<const __m256 *>(src)); }

	template<std::same_as<double> T, signed_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256d &src) noexcept { *reinterpret_cast<__m256i *>(dst) = cvt_f64_i64(src); }
	template<signed_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256i &src) noexcept { *reinterpret_cast<__m256d *>(dst) = cvt_i64_f64(src); }
	template<std::same_as<double> T, unsigned_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256d &src) noexcept { *reinterpret_cast<__m256i *>(dst) = cvt_f64_u64(src); }
	template<unsigned_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(U *dst, const __m256i &src) noexcept { *reinterpret_cast<__m256d *>(dst) = cvt_u64_f64(src); }

	template<std::same_as<double> T, signed_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(__m256d &dst, const U *src) noexcept { dst = cvt_i64_f64(*reinterpret_cast<const __m256i *>(src)); }
	template<signed_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(__m256i &dst, const U *src) noexcept { dst = cvt_f64_i64(*reinterpret_cast<const __m256d *>(src)); }
	template<std::same_as<double> T, unsigned_integral_of_size<8> U>
	DPM_FORCEINLINE void cast_copy(__m256d &dst, const U *src) noexcept { dst = cvt_u64_f64(*reinterpret_cast<const __m256i *>(src)); }
	template<unsigned_integral_of_size<8> T, std::same_as<double> U>
	DPM_FORCEINLINE void cast_copy(__m256i &dst, const U *src) noexcept { dst = cvt_f64_u64(*reinterpret_cast<const __m256d *>(src)); }
#endif

	template<typename V, typename T0, std::convertible_to<T0>... Ts>
	[[nodiscard]] DPM_FORCEINLINE V set(T0 v0, Ts... vs) noexcept
	{
		T0 vals[] = {v0, static_cast<T0>(vs)...};
		return set<V>(vals);
	}

	/* Some algorithms require a left-aligned mask instead of right-aligned. */
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask_l(V x, std::size_t n) noexcept
	{
		constexpr auto extent = sizeof(V) / sizeof(T);
		auto bits = detail::movemask<T>(x) << (std::numeric_limits<std::size_t>::digits - extent * movemask_bits_v<T>);
		if (n < extent) for (n = extent - n; n--;) bits <<= movemask_bits_v<T>;
		return bits;
	}
}