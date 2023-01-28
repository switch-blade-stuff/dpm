/*
 * Created by switch_blade on 2023-01-25.
 */

#pragma once

#include "../type_fwd.hpp"

namespace dpm::detail
{
	template<typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V fill(float value) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		return std::bit_cast<V>(_mm_set1_ps(std::bit_cast<float>(value)));
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

	template<typename... Ts, typename V>
	[[nodiscard]] DPM_FORCEINLINE V set(Ts... vs) noexcept requires (sizeof...(Ts) == 4 && sizeof(V) == 16)
	{
		return std::bit_cast<V>(_mm_set_ps(std::bit_cast<float>(vs)...));
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_ps(std::bit_cast<__m128>(x)));
	}
	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(bool *dst, const V &src, std::size_t n) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(src);
		for (std::size_t i = 0; i < n && i < 4; ++i)dst[i] = bits & (1 << i);
	}
	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(V &dst, const bool *src, std::size_t n) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		std::int32_t values[4] = {0};
		for (std::size_t i = 0; i < n && i < 4; ++i)values[i] = extend_bool<std::int32_t>(src[i]);
		dst = set<V>(values[3], values[2], values[1], values[0]);
	}
	template<typename T, typename V, typename F>
	DPM_FORCEINLINE void mask_invoke(V x, F fn, std::size_t n) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(x);
		for (std::size_t i = 0; i < n && i < 4; ++i)
			fn(i, bits & (1 << i));
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
#ifdef DPM_HAS_SSE4_1
		switch (n)
		{
			case 3: return std::bit_cast<V>(_mm_blend_ps(va, vb, 0b1000));
			case 2: return std::bit_cast<V>(_mm_blend_ps(va, vb, 0b1100));
			case 1: return std::bit_cast<V>(_mm_blend_ps(va, vb, 0b1110));
			default: return std::bit_cast<V>(a);
		}
#else
		auto vm = _mm_undefined_ps();
		switch (const auto mask = std::bit_cast<float>(0xffff'ffff); n)
		{
			case 3: vm = _mm_set_ps(mask, 0.0f, 0.0f, 0.0f); break;
			case 2: vm = _mm_set_ps(mask, mask, 0.0f, 0.0f); break;
			case 1: vm = _mm_set_ps(mask, mask, mask, 0.0f); break;
			default: return std::bit_cast<V>(a);
		}
		return std::bit_cast<V>(_mm_or_ps(_mm_andnot_ps(vm, va), _mm_and_ps(vm, vb)));
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto mask = std::bit_cast<float>(0xffff'ffff);
		const auto vx = std::bit_cast<__m128>(x);
		switch (n)
		{
			case 3: return std::bit_cast<V>(_mm_and_ps(vx, _mm_set_ps(0.0f, mask, mask, mask)));
			case 2: return std::bit_cast<V>(_mm_and_ps(vx, _mm_set_ps(0.0f, 0.0f, mask, mask)));
			case 1: return std::bit_cast<V>(_mm_and_ps(vx, _mm_set_ps(0.0f, 0.0f, 0.0f, mask)));
			default: return x;
		}
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16)

	template<typename T, typename V, typename M>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, M m) noexcept requires (sizeof(T) == 4 && sizeof(V) == 16 && sizeof(M) == 16)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		const auto vm = std::bit_cast<__m128>(m);
		return std::bit_cast<V>(_mm_blendv_ps(va, vb, vm));
	}

#ifdef DPM_HAS_SSE2
	template<typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept requires (sizeof(V) == 16 && sizeof(T) == 1)
	{
		return std::bit_cast<V>(_mm_set1_epi8(std::bit_cast<std::int8_t>(value)));
	}
	template<typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept requires (sizeof(V) == 16 && sizeof(T) == 2)
	{
		return std::bit_cast<V>(_mm_set1_epi16(std::bit_cast<std::int16_t>(value)));
	}
	template<typename V, typename T>
	[[nodiscard]] DPM_FORCEINLINE V fill(T value) noexcept requires (sizeof(V) == 16 && sizeof(T) == 8)
	{
		return std::bit_cast<V>(_mm_set1_pd(std::bit_cast<double>(value)));
	}

	template<typename... Ts, typename V>
	[[nodiscard]] DPM_FORCEINLINE V set(Ts... vs) noexcept requires (sizeof...(Ts) == 16 && sizeof(V) == 16)
	{
		return std::bit_cast<V>(_mm_set_epi8(std::bit_cast<std::int8_t>(vs)...));
	}
	template<typename... Ts, typename V>
	[[nodiscard]] DPM_FORCEINLINE V set(Ts... vs) noexcept requires (sizeof...(Ts) == 8 && sizeof(V) == 16)
	{
		return std::bit_cast<V>(_mm_set_epi16(std::bit_cast<std::int16_t>(vs)...));
	}
	template<typename... Ts, typename V>
	[[nodiscard]] DPM_FORCEINLINE V set(Ts... vs) noexcept requires (sizeof...(Ts) == 2 && sizeof(V) == 16)
	{
		return std::bit_cast<V>(_mm_set_pd(std::bit_cast<double>(vs)...));
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_epi8(std::bit_cast<__m128>(x)));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE std::size_t movemask(V x) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		return static_cast<std::size_t>(_mm_movemask_pd(std::bit_cast<__m128>(x)));
	}

	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(bool *dst, const V &src, std::size_t n) noexcept requires (sizeof(T) == 1 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(src);
		for (std::size_t i = 0; i < n && i < 16; ++i)
			dst[i] = bits & (1 << i);
	}
	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(bool *dst, const V &src, std::size_t n) noexcept requires (sizeof(T) == 2 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(src);
		for (std::size_t i = 0; i < n && i < 8; ++i)
			dst[i] = bits & (1 << i);
	}
	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(bool *dst, const V &src, std::size_t n) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(src);
		if (n > 1) dst[1] = bits & 2;
		if (n > 0) dst[0] = bits & 1;
	}

	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(V &dst, const bool *src, std::size_t n) noexcept requires (sizeof(T) == 1 && sizeof(V) == 16)
	{
		std::int8_t values[16] = {0};
		for (std::size_t i = 0; i < n && i < 16; ++i) values[i] = extend_bool<std::int8_t>(src[i]);
		dst = set<V>(
				values[15], values[14], values[13], values[12],
				values[11], values[10], values[9], values[8],
				values[7], values[6], values[5], values[4],
				values[3], values[2], values[1], values[0]
		);
	}
	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(V &dst, const bool *src, std::size_t n) noexcept requires (sizeof(T) == 2 && sizeof(V) == 16)
	{
		std::int16_t values[8] = {0};
		for (std::size_t i = 0; i < n && i < 4; ++i) values[i] = extend_bool<std::int16_t>(src[i]);
		dst = set<V>(values[7], values[6], values[5], values[4], values[3], values[2], values[1], values[0]);
	}
	template<typename T, typename V>
	DPM_FORCEINLINE void mask_copy(V &dst, const bool *src, std::size_t n) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		std::uint64_t values[2] = {0};
		if (n > 1) values[1] = extend_bool<std::int64_t>(src[1]);
		if (n > 0) values[0] = extend_bool<std::int64_t>(src[0]);
		dst = set<V>(values[1], values[0]);
	}

	template<typename T, typename V, typename F>
	DPM_FORCEINLINE void mask_invoke(V x, F fn, std::size_t n) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(x);
		for (std::size_t i = 0; i < n && i < 16 / sizeof(T); ++i)
			fn(i, bits & (1 << (i * sizeof(T))));
	}
	template<typename T, typename V, typename F>
	DPM_FORCEINLINE void mask_invoke(V x, F fn, std::size_t n) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		const auto bits = movemask<T>(x);
		if (n > 1) fn(1, bits & 2);
		if (n > 0) fn(0, bits & 1);
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires (sizeof(T) == 1 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
#ifdef DPM_HAS_SSE4_1
		switch (const auto m = static_cast<std::int8_t>(0x80); n)
		{
			case 15: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 14: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 13: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 12: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 11: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 10: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 9: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 8: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 7: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0)));
			case 6: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0)));
			case 5: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0)));
			case 4: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0)));
			case 3: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0)));
			case 2: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0)));
			case 1: return std::bit_cast<V>(_mm_blendv_epi8(va, vb, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0)));
			default: return a;
		}
#else
		auto vm = _mm_undefined_si128();
		switch (const auto m = static_cast<std::int8_t>(0xff); n)
		{
			case 15: vm = _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 14: vm = _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 13: vm = _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 12: vm = _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 11: vm = _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 10: vm = _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 9: vm = _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 8: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0); break;
			case 7: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0); break;
			case 6: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0); break;
			case 5: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0); break;
			case 4: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0); break;
			case 3: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0); break;
			case 2: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0); break;
			case 1: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0); break;
			default: return a;
		}
		return std::bit_cast<V>(_mm_or_si128(_mm_andnot_si128(vm, va), _mm_and_si128(vm, vb)));
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires (sizeof(T) == 1 && sizeof(V) == 16)
	{
		const auto m = static_cast<std::int8_t>(0xff);
		const auto vx = std::bit_cast<__m128i>(x);
		switch (n)
		{
			case 15: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m)));
			case 14: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, m, m, m, m, m, m, m, m, m, m, m, m, m, m)));
			case 13: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, m, m, m, m, m, m, m, m, m, m, m, m, m)));
			case 12: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m, m, m)));
			case 11: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m, m)));
			case 10: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m)));
			case 9: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m)));
			case 8: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m)));
			case 7: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m)));
			case 6: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m)));
			case 5: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m)));
			case 4: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m)));
			case 3: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m)));
			case 2: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m)));
			case 1: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m)));
			default: return x;
		}
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires (sizeof(T) == 1 && sizeof(V) == 16)
	{
		const auto m = static_cast<std::int8_t>(0xff);
		const auto vx = std::bit_cast<__m128i>(x);
		switch (n)
		{
			case 15: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 14: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 13: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 12: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 11: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 10: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 9: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 8: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0)));
			case 7: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0)));
			case 6: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0)));
			case 5: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0)));
			case 4: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0)));
			case 3: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0)));
			case 2: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0)));
			case 1: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0)));
			default: return x;
		}
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires (sizeof(T) == 2 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
#ifdef DPM_HAS_SSE4_1
		switch (n)
		{
			case 7: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1000'0000));
			case 6: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1100'0000));
			case 5: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1110'0000));
			case 4: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1111'0000));
			case 3: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1111'1000));
			case 2: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1111'1100));
			case 1: return std::bit_cast<V>(_mm_blend_epi16(va, vb, 0b1111'1110));
			default: return a;
		}
#else
		auto vm = _mm_undefined_si128();
		switch (const auto mask = static_cast<std::int16_t>(0xffff); n)
		{
			case 7: vm = _mm_set_epi16(m, 0, 0, 0, 0, 0, 0, 0); break;
			case 6: vm = _mm_set_epi16(m, m, 0, 0, 0, 0, 0, 0); break;
			case 5: vm = _mm_set_epi16(m, m, m, 0, 0, 0, 0, 0); break;
			case 4: vm = _mm_set_epi16(m, m, m, m, 0, 0, 0, 0); break;
			case 3: vm = _mm_set_epi16(m, m, m, m, m, 0, 0, 0); break;
			case 2: vm = _mm_set_epi16(m, m, m, m, m, m, 0, 0); break;
			case 1: vm = _mm_set_epi16(m, m, m, m, m, m, m, 0); break;
			default: return a;
		}
		return std::bit_cast<V>(_mm_or_si128(_mm_andnot_si128(vm, va), _mm_and_si128(vm, vb)));
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires (sizeof(T) == 2 && sizeof(V) == 16)
	{
		const auto m = static_cast<std::int16_t>(0xffff);
		const auto vx = std::bit_cast<__m128i>(x);
		switch (n)
		{
			case 7: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, m, m, m, m, m, m, m)));
			case 6: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, 0, m, m, m, m, m, m)));
			case 5: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, 0, 0, m, m, m, m, m)));
			case 4: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, 0, 0, 0, m, m, m, m)));
			case 3: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, 0, 0, 0, 0, m, m, m)));
			case 2: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, 0, 0, 0, 0, 0, m, m)));
			case 1: return std::bit_cast<V>(_mm_and_si128(vx, _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, m)));
			default: return x;
		}
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires (sizeof(T) == 2 && sizeof(V) == 16)
	{
		const auto m = static_cast<std::int16_t>(0xffff);
		const auto vx = std::bit_cast<__m128i>(x);
		switch (n)
		{
			case 7: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, 0, 0, 0, 0, 0, 0, 0)));
			case 6: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, m, 0, 0, 0, 0, 0, 0)));
			case 5: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, m, m, 0, 0, 0, 0, 0)));
			case 4: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, m, m, m, 0, 0, 0, 0)));
			case 3: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, m, m, m, m, 0, 0, 0)));
			case 2: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, m, m, m, m, m, 0, 0)));
			case 1: return std::bit_cast<V>(_mm_or_si128(vx, _mm_set_epi16(m, m, m, m, m, m, m, 0)));
			default: return x;
		}
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		if (n == 1)
		{
#ifdef DPM_HAS_SSE4_1
			return std::bit_cast<V>(_mm_blend_pd(va, vb, 0b10));
#else
			const auto vm = _mm_set_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff), 0.0);
			return std::bit_cast<V>(_mm_or_pd(_mm_andnot_pd(vm, va), _mm_and_pd(vm, vb)));
#endif
		}
		return va;
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		if (n == 1)
		{
			const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
			return std::bit_cast<V>(_mm_and_pd(std::bit_cast<__m128d>(x), _mm_set_pd(0.0, mask)));
		}
		return x;
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16)
	{
		if (n == 1)
		{
			const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
			return std::bit_cast<V>(_mm_or_pd(std::bit_cast<__m128d>(x), _mm_set_pd(mask, 0.0)));
		}
		return x;
	}

	template<typename T, typename V, typename M>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, M m) noexcept requires (sizeof(T) <= 2 && sizeof(V) == 16 && sizeof(M) == 16)
	{
		const auto va = std::bit_cast<__m128i>(a);
		const auto vb = std::bit_cast<__m128i>(b);
		const auto vm = std::bit_cast<__m128i>(m);
		return std::bit_cast<V>(_mm_blendv_epi8(va, vb, vm));
	}
	template<typename T, typename V, typename M>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, M m) noexcept requires (sizeof(T) == 8 && sizeof(V) == 16 && sizeof(M) == 16)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		const auto vm = std::bit_cast<__m128d>(m);
		return std::bit_cast<V>(_mm_blendv_pd(va, vb, vm));
	}
#endif
}