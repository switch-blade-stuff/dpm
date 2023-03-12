/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "type_fwd.hpp"
#include "cvt.hpp"

namespace dpm::detail
{
	/* In certain cases, CLang has been generating cvtsi2ss + broadcast instructions for setzero. Use manual xor([xy]mm, [xy]mm) to avoid that. */
	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setzero() noexcept requires (sizeof(V) == 16)
	{
#ifdef __clang__
		const auto tmp = _mm_undefined_ps();
		return std::bit_cast<V>(_mm_xor_ps(tmp, tmp));
#else
		return std::bit_cast<V>(_mm_setzero_ps());
#endif
	}
	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setones() noexcept requires (sizeof(V) == 16)
	{
#if !defined(DPM_HAS_SSE2) || !defined(__clang__)
		const auto tmp = setzero<__m128>();
		return std::bit_cast<V>(_mm_cmpeq_ps(tmp, tmp));
#else
		const auto tmp = _mm_undefined_si128();
		return std::bit_cast<V>(_mm_cmpeq_epi32(tmp, tmp));
#endif
	}

	template<std::same_as<__m128> V>
	[[nodiscard]] DPM_FORCEINLINE V set(const float (&vals)[4]) noexcept { return _mm_set_ps(vals[3], vals[2], vals[1], vals[0]); }
	template<std::same_as<__m128> V>
	[[nodiscard]] DPM_FORCEINLINE V fill(auto value) noexcept { return _mm_set1_ps(value); }
	template<std::same_as<__m128> V>
	[[nodiscard]] DPM_FORCEINLINE V undefined() noexcept { return _mm_undefined_ps(); }

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
	[[nodiscard]] DPM_FORCEINLINE V fill(auto value) noexcept { return _mm_set1_pd(value); }
	template<std::same_as<__m128i> V, integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi8(static_cast<std::int8_t>(value)); }
	template<std::same_as<__m128i> V, integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi16(static_cast<std::int16_t>(value)); }
	template<std::same_as<__m128i> V, integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi32(static_cast<std::int32_t>(value)); }
	template<std::same_as<__m128i> V, integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm_set1_epi64x(static_cast<std::int64_t>(value)); }

	template<std::same_as<__m128d> V>
	[[nodiscard]] DPM_FORCEINLINE V undefined() noexcept { return _mm_undefined_pd(); }
	template<std::same_as<__m128i> V>
	[[nodiscard]] DPM_FORCEINLINE V undefined() noexcept { return _mm_undefined_si128(); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_epi8(std::bit_cast<__m128i>(x)));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_pd(std::bit_cast<__m128d>(x)));
	}

	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstoreu(T *dst, V src, M mask) noexcept
	{
		const auto data = reinterpret_cast<alias_t<char> *>(dst);
		_mm_maskmoveu_si128(std::bit_cast<__m128i>(src), std::bit_cast<__m128i>(mask), data);
	}
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
#ifdef __clang__
		const auto tmp = _mm256_undefined_ps();
		return std::bit_cast<V>(_mm256_xor_ps(tmp, tmp));
#else
		return std::bit_cast<V>(_mm256_setzero_ps());
#endif
	}
	template<typename V>
	[[nodiscard]] DPM_FORCEINLINE V setones() noexcept requires (sizeof(V) == 32)
	{
#if !defined(DPM_HAS_AVX2) || !defined(__clang__)
		const auto tmp = setzero<__m256>();
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
	[[nodiscard]] DPM_FORCEINLINE V fill(auto value) noexcept { return _mm256_set1_ps(value); }
	template<std::same_as<__m256d> V>
	[[nodiscard]] DPM_FORCEINLINE V fill(auto value) noexcept { return _mm256_set1_pd(value); }
	template<std::same_as<__m256i> V, integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi8(static_cast<std::int8_t>(value)); }
	template<std::same_as<__m256i> V, integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi16(static_cast<std::int16_t>(value)); }
	template<std::same_as<__m256i> V, integral_of_size<4> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi32(static_cast<std::int32_t>(value)); }
	template<std::same_as<__m256i> V, integral_of_size<8> T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept { return _mm256_set1_epi64x(static_cast<std::int64_t>(value)); }

	template<std::same_as<__m256> V>
	[[nodiscard]] DPM_FORCEINLINE V undefined() noexcept { return _mm256_undefined_ps(); }
	template<std::same_as<__m256d> V>
	[[nodiscard]] DPM_FORCEINLINE V undefined() noexcept { return _mm256_undefined_pd(); }
	template<std::same_as<__m256i> V>
	[[nodiscard]] DPM_FORCEINLINE V undefined() noexcept { return _mm256_undefined_si256(); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 32)
	{
#ifdef DPM_HAS_AVX2
		return static_cast<std::size_t>(_mm256_movemask_epi8(std::bit_cast<__m256i>(x)));
#else
		const auto ml = movemask<T>(_mm256_extractf128_si256(std::bit_cast<__m256i>(x), 0));
		const auto mh = movemask<T>(_mm256_extractf128_si256(std::bit_cast<__m256i>(x), 1));
		return (mh << ((16 / sizeof(T)) * movemask_bits_v<T>)) | ml;
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

	template<typename To, typename From, typename VTo, typename VFrom>
	DPM_FORCEINLINE void cast_copy(VTo &dst, const VFrom &src) noexcept requires (sizeof(VTo) == 16 && sizeof(VFrom) == 32) { dst = cvt<To, From>(src); }

	template<typename V, typename T>
	DPM_FORCEINLINE V maskload(const T *src, __m128i mask) noexcept requires (sizeof(T) == 4)
	{
		const auto data = reinterpret_cast<const alias_t<float> *>(src);
		return std::bit_cast<V>(_mm_maskload_ps(data, mask));
	}
	template<typename V, typename T>
	DPM_FORCEINLINE V maskload(const T *src, __m128i mask) noexcept requires (sizeof(T) == 8)
	{
		const auto data = reinterpret_cast<const alias_t<double> *>(src);
		return std::bit_cast<V>(_mm_maskload_pd(data, mask));
	}
	template<typename V, typename T>
	DPM_FORCEINLINE V maskload(const T *src, __m256i mask) noexcept requires (sizeof(T) == 4)
	{
		const auto data = reinterpret_cast<const alias_t<float> *>(src);
		return std::bit_cast<V>(_mm256_maskload_ps(data, mask));
	}
	template<typename V, typename T>
	DPM_FORCEINLINE V maskload(const T *src, __m256i mask) noexcept requires (sizeof(T) == 8)
	{
		const auto data = reinterpret_cast<const alias_t<double> *>(src);
		return std::bit_cast<V>(_mm256_maskload_pd(data, mask));
	}

	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstore(T *dst, V src, M mask) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto data = reinterpret_cast<alias_t<float> *>(dst);
		_mm_maskstore_ps(data, std::bit_cast<__m128i>(mask), std::bit_cast<__m128>(src));
	}
	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstore(T *dst, V src, M mask) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		const auto data = reinterpret_cast<alias_t<double> *>(dst);
		_mm_maskstore_pd(data, std::bit_cast<__m128i>(mask), std::bit_cast<__m128d>(src));
	}
	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstore(T *dst, V src, M mask) noexcept requires (sizeof(T) == 4 && sizeof(V) == 32)
	{
		const auto data = reinterpret_cast<alias_t<float> *>(dst);
		_mm256_maskstore_ps(data, std::bit_cast<__m256i>(mask), std::bit_cast<__m256>(src));
	}
	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstore(T *dst, V src, M mask) noexcept requires (sizeof(T) == 8 && sizeof(V) == 32)
	{
		const auto data = reinterpret_cast<alias_t<double> *>(dst);
		_mm256_maskstore_pd(data, std::bit_cast<__m256i>(mask), std::bit_cast<__m256d>(src));
	}
#elif defined(DPM_HAS_SSE2)
	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstore(T *dst, V src, M mask) noexcept requires (sizeof(V) == 16) { maskstoreu(dst, src, mask); }
	template<typename V, typename T, typename M>
	DPM_FORCEINLINE void maskstore(T *dst, V src, M mask) noexcept requires (sizeof(V) == 32)
	{
		mux_128x2<__m256i>([&](auto v, auto m) { maskstoreu(dst + (32 / sizeof(T)), v, m); }, src, mask);
	}
#endif

	template<typename T, typename V, typename... Args, typename F, typename... Vs>
	DPM_FORCEINLINE void mask_invoke(F f, std::size_t m, Vs &&...args) noexcept requires (sizeof...(Args) == sizeof...(Vs))
	{
		for (std::size_t i = 0; i < sizeof(V) / sizeof(T) && m; ++i, m >>= 1)
		{
			if ((m & 1) == 0) continue;
			f(reinterpret_cast<alias_t<std::remove_reference_t<Args>> *>(&args)[i]...);
		}
	}

	template<typename To, typename From, typename VTo, typename VFrom>
	DPM_FORCEINLINE void cast_copy(VTo &dst, const VFrom &src) noexcept requires (!std::same_as<To, From> && !(std::integral<To> && std::integral<From> && sizeof(To) == sizeof(From)))
	{
		dst = cvt<To, From>(src);
	}
	template<typename To, typename From, typename VTo, typename VFrom>
	DPM_FORCEINLINE void cast_copy(VTo &dst, const VFrom &src) noexcept requires (std::same_as<To, From> || (std::integral<To> && std::integral<From> && sizeof(To) == sizeof(From)))
	{
		dst = src;
	}
	template<typename To, typename From, typename VFrom>
	DPM_FORCEINLINE void cast_copy(To *dst, const VFrom &src) noexcept
	{
		using dst_vector = select_vector_t<To, sizeof(VFrom)>;
		cast_copy<To, From>(*reinterpret_cast<dst_vector *>(dst), src);
	}
	template<typename To, typename From, typename VTo>
	DPM_FORCEINLINE void cast_copy(VTo &dst, const From *src) noexcept
	{
		using src_vector = select_vector_t<From, sizeof(VTo)>;
		cast_copy<To, From>(dst, *reinterpret_cast<const src_vector *>(src));
	}

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
		constexpr std::size_t extent = sizeof(V) / sizeof(T);
		auto bits = detail::movemask<T>(x) << (std::numeric_limits<std::size_t>::digits - extent * movemask_bits_v<T>);
		if (n < extent) for (n = extent - n; n--;) bits <<= movemask_bits_v<T>;
		return bits;
	}
}