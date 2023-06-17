/*
 * Created by switchblade on 2023-01-29.
 */

#pragma once

#include "utility.hpp"

namespace dpm::detail
{
	template<std::size_t I3, std::size_t I2, std::size_t I1, std::size_t I0>
	[[nodiscard]] constexpr int shuffle_mask() noexcept { return _MM_SHUFFLE(I3, I2, I1, I0); }

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires(sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
#ifdef DPM_HAS_SSE4_1
		switch (n)
		{
			case 3: return std::bit_cast<V>(_mm_blend_ps(va, vb, 0b1000));
			case 2: return std::bit_cast<V>(_mm_blend_ps(va, vb, 0b1100));
			case 1: return std::bit_cast<V>(_mm_blend_ps(va, vb, 0b1110));
			default: return a;
		}
#else
		auto vm = _mm_undefined_ps();
		switch (const auto mask = std::bit_cast<float>(0xffff'ffff); n)
		{
		case 3: vm = _mm_set_ps(mask, 0.0f, 0.0f, 0.0f); break;
		case 2: vm = _mm_set_ps(mask, mask, 0.0f, 0.0f); break;
		case 1: vm = _mm_set_ps(mask, mask, mask, 0.0f); break;
		default: return a;
		}
		return std::bit_cast<V>(_mm_or_ps(_mm_andnot_ps(vm, va), _mm_and_ps(vm, vb)));
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires(sizeof(T) == 4 && sizeof(V) == 16)
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
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires(sizeof(T) == 4 && sizeof(V) == 16)
	{
		const auto mask = std::bit_cast<float>(0xffff'ffff);
		const auto vx = std::bit_cast<__m128>(x);
		switch (n)
		{
			case 3: return std::bit_cast<V>(_mm_or_ps(vx, _mm_set_ps(mask, 0.0f, 0.0f, 0.0f)));
			case 2: return std::bit_cast<V>(_mm_or_ps(vx, _mm_set_ps(mask, mask, 0.0f, 0.0f)));
			case 1: return std::bit_cast<V>(_mm_or_ps(vx, _mm_set_ps(mask, mask, mask, 0.0f)));
			default: return x;
		}
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, V m) noexcept requires(sizeof(T) == 4 && sizeof(V) == 16)
	{
#ifdef DPM_HAS_SSE4_1
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		const auto vm = std::bit_cast<__m128>(m);
		return std::bit_cast<V>(_mm_blendv_ps(va, vb, vm));
#else
		const auto va = std::bit_cast<__m128>(a);
		const auto vb = std::bit_cast<__m128>(b);
		const auto vm = std::bit_cast<__m128>(m);
		return std::bit_cast<V>(_mm_or_ps(_mm_andnot_ps(vm, va), _mm_and_ps(vm, vb)));
#endif
	}

	template<typename T, typename V, std::size_t I, std::size_t... Is>
	concept sequence_shuffle = ((I / (sizeof(V) / sizeof(T)) == Is / (sizeof(V) / sizeof(T))) && ...) && is_sequential<-1, I, I, Is...>::value;

	/* If all indices are from the same vector, no shuffle is needed. */
	template<typename T, std::size_t I, std::size_t... Is, typename V>
	[[nodiscard]] DPM_FORCEINLINE V shuffle(std::index_sequence<I, Is...>, const V *x) noexcept requires sequence_shuffle<T, V, I, Is...> { return x[I / (sizeof(V) / sizeof(T))]; }

	template<typename T, std::size_t I3, std::size_t I2, std::size_t I1, std::size_t I0, typename V>
	[[nodiscard]] DPM_FORCEINLINE V shuffle(std::index_sequence<I3, I2, I1, I0>, const V *x) noexcept requires(!sequence_shuffle<T, V, I3, I2, I1, I0> && sizeof(T) == 4 && sizeof(V) == 16)
	{
		constexpr auto P0 = I0 / 4, P1 = I1 / 4, P2 = I2 / 4, P3 = I3 / 4;
		const auto va = std::bit_cast<__m128>(x[P0]);
		const auto vb = std::bit_cast<__m128>(x[P2]);
		if constexpr (P0 == P1 && P2 == P3)
			return std::bit_cast<V>(_mm_shuffle_ps(va, vb, _MM_SHUFFLE(I3 % 4, I2 % 4, I1 % 4, I0 % 4)));
		else
		{
			const auto a = _mm_shuffle_ps(std::bit_cast<__m128>(x[P0]), std::bit_cast<__m128>(x[P1]), _MM_SHUFFLE(I1 % 4, I1 % 4, I0 % 4, I0 % 4));
			const auto b = _mm_shuffle_ps(std::bit_cast<__m128>(x[P2]), std::bit_cast<__m128>(x[P3]), _MM_SHUFFLE(I3 % 4, I3 % 4, I2 % 4, I2 % 4));
			return std::bit_cast<V>(_mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0)));
		}
	}

	template<std::same_as<float> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m128 v, Op op) noexcept
	{
#ifndef DPM_HAS_SSE3
		const auto a = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 1, 1));
#else
		const auto a = _mm_movehdup_ps(v);
#endif
		const auto b = op(v, a);
		return _mm_cvtss_f32(op(b, _mm_movehl_ps(a, b)));
	}

#ifdef DPM_HAS_SSE2
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i maskblend(__m128i a, __m128i b, std::size_t n) noexcept
	{
#ifdef DPM_HAS_SSE4_1
		switch (const auto m = static_cast<std::int8_t>(0x80); n)
		{
			case 15: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 14: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 13: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 12: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 11: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 10: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 9: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 8: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0));
			case 7: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0));
			case 6: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0));
			case 5: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0));
			case 4: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0));
			case 3: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0));
			case 2: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0));
			case 1: return _mm_blendv_epi8(a, b, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0));
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
		return _mm_or_si128(_mm_andnot_si128(vm, a), _mm_and_si128(vm, b));
#endif
	}
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i maskblend(__m128i a, __m128i b, std::size_t n) noexcept
	{
#ifdef DPM_HAS_SSE4_1
		switch (n)
		{
			case 7: return _mm_blend_epi16(a, b, 0b1000'0000);
			case 6: return _mm_blend_epi16(a, b, 0b1100'0000);
			case 5: return _mm_blend_epi16(a, b, 0b1110'0000);
			case 4: return _mm_blend_epi16(a, b, 0b1111'0000);
			case 3: return _mm_blend_epi16(a, b, 0b1111'1000);
			case 2: return _mm_blend_epi16(a, b, 0b1111'1100);
			case 1: return _mm_blend_epi16(a, b, 0b1111'1110);
			default: return a;
		}
#else
		auto vm = _mm_undefined_si128();
		switch (const auto m = static_cast<std::int16_t>(0xffff); n)
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
		return _mm_or_si128(_mm_andnot_si128(vm, a), _mm_and_si128(vm, b));
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires(sizeof(T) == 8 && sizeof(V) == 16)
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

	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i maskzero(__m128i x, std::size_t n) noexcept
	{
		const auto m = static_cast<std::int8_t>(0xff);
		switch (n)
		{
			case 15: return _mm_and_si128(x, _mm_set_epi8(0, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m));
			case 14: return _mm_and_si128(x, _mm_set_epi8(0, 0, m, m, m, m, m, m, m, m, m, m, m, m, m, m));
			case 13: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, m, m, m, m, m, m, m, m, m, m, m, m, m));
			case 12: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m, m, m));
			case 11: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m, m));
			case 10: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m));
			case 9: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m));
			case 8: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m));
			case 7: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m));
			case 6: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m));
			case 5: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m));
			case 4: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m));
			case 3: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m));
			case 2: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m));
			case 1: return _mm_and_si128(x, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m));
			default: return x;
		}
	}
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i maskzero(__m128i x, std::size_t n) noexcept
	{
		const auto m = static_cast<std::int16_t>(0xffff);
		switch (n)
		{
			case 7: return _mm_and_si128(x, _mm_set_epi16(0, m, m, m, m, m, m, m));
			case 6: return _mm_and_si128(x, _mm_set_epi16(0, 0, m, m, m, m, m, m));
			case 5: return _mm_and_si128(x, _mm_set_epi16(0, 0, 0, m, m, m, m, m));
			case 4: return _mm_and_si128(x, _mm_set_epi16(0, 0, 0, 0, m, m, m, m));
			case 3: return _mm_and_si128(x, _mm_set_epi16(0, 0, 0, 0, 0, m, m, m));
			case 2: return _mm_and_si128(x, _mm_set_epi16(0, 0, 0, 0, 0, 0, m, m));
			case 1: return _mm_and_si128(x, _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, m));
			default: return x;
		}
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires(sizeof(T) == 8 && sizeof(V) == 16)
	{
		if (n == 1)
		{
			const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
			return std::bit_cast<V>(_mm_and_pd(std::bit_cast<__m128d>(x), _mm_set_pd(0.0, mask)));
		}
		return x;
	}

	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i maskone(__m128i x, std::size_t n) noexcept
	{
		const auto m = static_cast<std::int8_t>(0xff);
		switch (n)
		{
			case 15: return _mm_or_si128(x, _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 14: return _mm_or_si128(x, _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 13: return _mm_or_si128(x, _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 12: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 11: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 10: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 9: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 8: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0));
			case 7: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0));
			case 6: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0));
			case 5: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0));
			case 4: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0));
			case 3: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0));
			case 2: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0));
			case 1: return _mm_or_si128(x, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0));
			default: return x;
		}
	}
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m128i maskone(__m128i x, std::size_t n) noexcept
	{
		const auto m = static_cast<std::int16_t>(0xffff);
		switch (n)
		{
			case 7: return _mm_or_si128(x, _mm_set_epi16(m, 0, 0, 0, 0, 0, 0, 0));
			case 6: return _mm_or_si128(x, _mm_set_epi16(m, m, 0, 0, 0, 0, 0, 0));
			case 5: return _mm_or_si128(x, _mm_set_epi16(m, m, m, 0, 0, 0, 0, 0));
			case 4: return _mm_or_si128(x, _mm_set_epi16(m, m, m, m, 0, 0, 0, 0));
			case 3: return _mm_or_si128(x, _mm_set_epi16(m, m, m, m, m, 0, 0, 0));
			case 2: return _mm_or_si128(x, _mm_set_epi16(m, m, m, m, m, m, 0, 0));
			case 1: return _mm_or_si128(x, _mm_set_epi16(m, m, m, m, m, m, m, 0));
			default: return x;
		}
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires(sizeof(T) == 8 && sizeof(V) == 16)
	{
		if (n == 1)
		{
			const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
			return std::bit_cast<V>(_mm_or_pd(std::bit_cast<__m128d>(x), _mm_set_pd(mask, 0.0)));
		}
		return x;
	}

	template<typename T>
	[[nodiscard]] DPM_FORCEINLINE __m128i blendv(__m128i a, __m128i b, __m128i m) noexcept requires(sizeof(T) <= 2)
	{
#ifdef DPM_HAS_SSE4_1
		return _mm_blendv_epi8(a, b, m);
#else
		return _mm_or_si128(_mm_andnot_si128(m, a), _mm_and_si128(m, b));
#endif
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, V m) noexcept requires(sizeof(T) == 8 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128d>(a);
		const auto vb = std::bit_cast<__m128d>(b);
		const auto vm = std::bit_cast<__m128d>(m);
#ifdef DPM_HAS_SSE4_1
		return std::bit_cast<V>(_mm_blendv_pd(va, vb, vm));
#else
		return std::bit_cast<V>(_mm_or_pd(_mm_andnot_pd(vm, va), _mm_and_pd(vm, vb)));
#endif
	}

	template<typename T, std::size_t I1, std::size_t I0, typename V>
	[[nodiscard]] DPM_FORCEINLINE V shuffle(std::index_sequence<I1, I0>, const V *x) noexcept requires(!sequence_shuffle<T, V, I1, I0> && sizeof(T) == 8 && sizeof(V) == 16)
	{
		const auto va = std::bit_cast<__m128d>(x[I0 / 2]);
		const auto vb = std::bit_cast<__m128d>(x[I1 / 2]);
		return std::bit_cast<V>(_mm_shuffle_pd(va, vb, _MM_SHUFFLE2(I1 % 2, I0 % 2)));
	}

	template<std::same_as<double> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m128d v, Op op) noexcept
	{
		const auto a = _mm_shuffle_pd(v, v, _MM_SHUFFLE2(1, 1));
		return _mm_cvtsd_f64(op(v, a));
	}
	template<integral_of_size<4> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m128i v, Op op) noexcept
	{
		const auto vf = std::bit_cast<__m128>(v);
#ifndef DPM_HAS_SSE3
		const auto a = _mm_shuffle_ps(vf, vf, _MM_SHUFFLE(3, 3, 1, 1));
#else
		const auto a = _mm_movehdup_ps(vf);
#endif
		const auto b = op(v, std::bit_cast<__m128i>(a));
		const auto c = _mm_movehl_ps(a, std::bit_cast<__m128>(b));
		return _mm_cvtsi128_si32(op(b, std::bit_cast<__m128i>(c)));
	}
	template<integral_of_size<8> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m128i v, Op op) noexcept
	{
		const auto vf = std::bit_cast<__m128d>(v);
		const auto a = _mm_shuffle_pd(vf, vf, _MM_SHUFFLE2(1, 1));
		return _mm_cvtsi128_si64x(op(v, std::bit_cast<__m128i>(a)));
	}
#endif

#ifdef DPM_HAS_SSSE3
	template<integral_of_size<1> T, std::size_t I, std::size_t... Is>
	[[nodiscard]] DPM_FORCEINLINE __m128i shuffle(std::index_sequence<I, Is...>, const __m128i *x) noexcept requires(!sequence_shuffle<T, __m128i, I, Is...>);
	template<std::size_t IA, std::size_t... IAs, std::size_t IB, std::size_t... IBs>
	[[nodiscard]] DPM_FORCEINLINE __m128i shuffle16pairs(std::index_sequence<IA, IAs...>, std::index_sequence<IB, IBs...>, const __m128i *x) noexcept
	{
		if constexpr (((IA / 4 == IAs / 4) && ...) && ((IB / 4 == IBs / 4) && ...) && IA / 8 != IB / 8)
		{
			constexpr auto J0 = (IA / 4) & 1, J1 = (IB / 4) & 1;
			constexpr auto ma = shuffle_mask<IA % 4, (IAs % 4)...>();
			constexpr auto mb = shuffle_mask<IB % 4, (IBs % 4)...>();
			__m128i a, b;

			if constexpr (J0)
				a = _mm_shufflehi_epi16(x[IA / 8], ma);
			else
				a = _mm_shufflelo_epi16(x[IA / 8], ma);
			if constexpr (J1)
				b = _mm_shufflehi_epi16(x[IB / 8], mb);
			else
				b = _mm_shufflelo_epi16(x[IB / 8], mb);
			return std::bit_cast<__m128i>(_mm_shuffle_pd(std::bit_cast<__m128d>(b), std::bit_cast<__m128d>(a), _MM_SHUFFLE2(J0, J1)));
		}
		else
			return shuffle<std::int8_t>(repeat_sequence_t<2, 1, IA, IAs..., IB, IBs...>{}, x);
	}
	template<integral_of_size<1> T, std::size_t I, std::size_t... Is>
	[[nodiscard]] DPM_FORCEINLINE __m128i shuffle(std::index_sequence<I, Is...>, const __m128i *x) noexcept requires(!sequence_shuffle<T, __m128i, I, Is...>)
	{
		__m128i result;
		if constexpr (!((I / 16 == Is / 16) && ...))
			shuffle_elements(reverse_sequence_t<I, Is...>{}, reinterpret_cast<alias_t<std::int8_t> *>(&result), reinterpret_cast<const alias_t<std::int8_t> *>(x));
		else
			result = _mm_shuffle_epi8(x[I / 16], _mm_set_epi8(I % 16, (Is % 16)...));
		return result;
	}
	template<integral_of_size<2> T, std::size_t I, std::size_t... Is>
	[[nodiscard]] DPM_FORCEINLINE __m128i shuffle(std::index_sequence<I, Is...>, const __m128i *x) noexcept requires(!sequence_shuffle<T, __m128i, I, Is...>)
	{
		return shuffle16pairs(extract_sequence_t<0, 4, I, Is...>{}, extract_sequence_t<4, 4, I, Is...>{}, x);
	}

	template<integral_of_size<1> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m128i v, Op op) noexcept
	{
		auto a = _mm_shuffle_epi8(v, _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 8));
		auto b = op(v, a);
		a = _mm_shuffle_epi8(b, _mm_set_epi8(7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4));
		b = op(b, a);
		a = _mm_shuffle_epi8(b, _mm_set_epi8(3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2));
		b = op(b, a);
		a = _mm_shuffle_epi8(b, _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
		return static_cast<T>(_mm_cvtsi128_si32(op(b, a)) & 0xff);
	}
	template<integral_of_size<2> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m128i v, Op op) noexcept
	{
		auto a = _mm_shuffle_epi8(v, _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 8));
		auto b = op(v, a);
		a = _mm_shuffle_epi8(b, _mm_set_epi8(7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4, 7, 6, 5, 4));
		b = op(b, a);
		a = _mm_shuffle_epi8(b, _mm_set_epi8(3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2));
		return static_cast<T>(_mm_cvtsi128_si32(op(b, a)) & 0xffff);
	}
#endif

#ifdef DPM_HAS_AVX
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires(sizeof(T) == 4 && sizeof(V) == 32)
	{
		const auto va = std::bit_cast<__m256>(a);
		const auto vb = std::bit_cast<__m256>(b);
		switch (n)
		{
			case 7: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1000'0000));
			case 6: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1100'0000));
			case 5: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1110'0000));
			case 4: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1111'0000));
			case 3: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1111'1000));
			case 2: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1111'1100));
			case 1: return std::bit_cast<V>(_mm256_blend_ps(va, vb, 0b1111'1110));
			default: return a;
		}
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskblend(V a, V b, std::size_t n) noexcept requires(sizeof(T) == 8 && sizeof(V) == 32)
	{
		const auto va = std::bit_cast<__m256d>(a);
		const auto vb = std::bit_cast<__m256d>(b);
		switch (n)
		{
			case 3: return std::bit_cast<V>(_mm256_blend_pd(va, vb, 0b1000));
			case 2: return std::bit_cast<V>(_mm256_blend_pd(va, vb, 0b1100));
			case 1: return std::bit_cast<V>(_mm256_blend_pd(va, vb, 0b1110));
			default: return a;
		}
	}

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, V m) noexcept requires(sizeof(T) == 4 && sizeof(V) == 32)
	{
		const auto va = std::bit_cast<__m256>(a);
		const auto vb = std::bit_cast<__m256>(b);
		const auto vm = std::bit_cast<__m256>(m);
		return std::bit_cast<V>(_mm256_blendv_ps(va, vb, vm));
	}
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V blendv(V a, V b, V m) noexcept requires(sizeof(T) == 8 && sizeof(V) == 32)
	{
		const auto va = std::bit_cast<__m256d>(a);
		const auto vb = std::bit_cast<__m256d>(b);
		const auto vm = std::bit_cast<__m256d>(m);
		return std::bit_cast<V>(_mm256_blendv_pd(va, vb, vm));
	}

	template<std::same_as<float> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m256 v, Op op) noexcept
	{
		auto a = _mm256_permute2f128_ps(v, v, 0b1000'0001);
		auto b = op(v, a);
		a = _mm256_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));
		b = op(b, a);
		return _mm256_cvtss_f32(op(b, _mm256_unpackhi_ps(b, b)));
	}
	template<std::same_as<double> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m256d v, Op op) noexcept
	{
		auto a = _mm256_permute2f128_pd(v, v, 0b1000'0001);
		auto b = op(v, a);
		return _mm256_cvtsd_f64(op(b, _mm256_unpackhi_pd(b, b)));
	}

#ifdef DPM_HAS_AVX2
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i maskblend(__m256i a, __m256i b, std::size_t n) noexcept
	{
		switch (const auto m = static_cast<std::int8_t>(0x80); n)
		{
			case 31: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 30: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 29: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 28: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 27: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 26: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 25: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 24: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 23: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 22: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 21: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 20: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 19: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 18: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 17: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 16: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 15: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 14: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 13: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 12: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 11: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 10: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 9: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 8: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0));
			case 7: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0));
			case 6: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0));
			case 5: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0));
			case 4: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0));
			case 3: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0));
			case 2: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0));
			case 1: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0));
			default: return a;
		}
	}
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i maskblend(__m256i a, __m256i b, std::size_t n) noexcept
	{
		switch (const auto m = static_cast<std::int8_t>(0x80); n)
		{
			case 15: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 14: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 13: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 12: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 11: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 10: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 9: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 8: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 7: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 6: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 5: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
			case 4: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0));
			case 3: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0));
			case 2: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0));
			case 1: return _mm256_blendv_epi8(a, b, _mm256_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0));
			default: return a;
		}
	}
	template<std::integral T>
	[[nodiscard]] DPM_FORCEINLINE __m256i blendv(__m256i a, __m256i b, __m256i m) noexcept
	{
		return _mm256_blendv_epi8(a, b, m);
	}

	template<integral_of_size<4> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m256i v, Op op) noexcept
	{
		auto a = _mm256_permute2f128_si256(v, v, 0b1000'0001);
		auto b = op(v, a);
		a = _mm256_shuffle_epi32(b, _MM_SHUFFLE(3, 3, 1, 1));
		b = op(b, a);
		return static_cast<T>(_mm256_cvtsi256_si32(op(b, _mm256_unpackhi_epi32(b, b))));
	}
	template<integral_of_size<8> T, typename Op>
	[[nodiscard]] DPM_FORCEINLINE T reduce(__m256i v, Op op) noexcept
	{
		auto a = _mm256_permute2f128_si256(v, v, 0b1000'0001);
		auto b = op(v, a);
		a = _mm256_unpackhi_epi64(b, b);
		b = op(b, a);
		return std::bit_cast<T>(_mm256_cvtsd_f64(std::bit_cast<__m256d>(b)));
	}
#else
	template<integral_of_size<1> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i maskblend(__m256i a, __m256i b, std::size_t n) noexcept
	{
		const auto h = maskblend<T>(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1), n - 16);
		const auto l = maskblend<T>(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(b, 0), n);
		return _mm256_set_m128i(h, l);
	}
	template<integral_of_size<2> T>
	[[nodiscard]] DPM_FORCEINLINE __m256i maskblend(__m256i a, __m256i b, std::size_t n) noexcept
	{
		const auto h = maskblend<T>(_mm256_extractf128_si256(a, 1), _mm256_extractf128_si256(b, 1), n - 8);
		const auto l = maskblend<T>(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(b, 0), n);
		return _mm256_set_m128i(h, l);
	}
	template<std::integral T>
	[[nodiscard]] DPM_FORCEINLINE __m256i blendv(__m256i a, __m256i b, __m256i m) noexcept
	{
		return mux_128x2<__m256i>([](auto a, auto b, auto m) { return blendv<T>(a, b, m); }, a, b, m);
	}
#endif

	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskzero(V x, std::size_t n) noexcept requires(sizeof(V) == 32) { return maskblend<T>(x, setzero<V>(), n); }
	template<typename T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V maskone(V x, std::size_t n) noexcept requires(sizeof(V) == 32) { return maskblend<T>(x, setones<V>(), n); }

	template<typename T, std::size_t I, std::size_t... Is, typename V>
	[[nodiscard]] DPM_FORCEINLINE V shuffle(std::index_sequence<I, Is...>, const V *x) noexcept requires(!sequence_shuffle<T, __m256i, I, Is...> && sizeof(V) == 32)
	{
		/* Since there are no *convenient* element-wise shuffles with AVX, use 2 SSE shuffles instead. */
		constexpr auto extent = 16 / sizeof(T);
		const auto x128 = reinterpret_cast<const select_vector_t<T, 16> *>(x);
		const auto l = std::bit_cast<__m128>(shuffle<T>(extract_sequence_t<extent, extent, I, Is...>{}, x128));
		const auto h = std::bit_cast<__m128>(shuffle<T>(extract_sequence_t<0, extent, I, Is...>{}, x128));
		return std::bit_cast<V>(_mm256_set_m128(h, l));
	}
#endif

	template<typename T, std::size_t J, std::size_t... Is, typename VTo, typename VFrom>
	DPM_FORCEINLINE void shuffle(std::index_sequence<Is...> is, VTo *dst, const VFrom *src) noexcept
	{
		if constexpr (alignof(VTo) > alignof(VFrom)) return shuffle<T, J>(is, reinterpret_cast<VFrom *>(dst), src);

		constexpr auto native_extent = sizeof(VTo) / sizeof(T);
		constexpr auto next_pos = (J + 1) * native_extent;
		constexpr auto base_pos = J * native_extent;
		if constexpr (sizeof...(Is) == native_extent)
			dst[J] = shuffle<T>(reverse_sequence_t<Is...>{}, reinterpret_cast<const VTo *>(src));
		else if constexpr (sizeof...(Is) > native_extent)
		{
			shuffle<T, J>(extract_sequence_t<base_pos, native_extent, Is...>{}, dst, src);
			shuffle<T, J>(extract_sequence_t<next_pos, sizeof...(Is) - native_extent, Is...>{}, dst + 1, src);
		}
		else
			shuffle<T, J>(pad_sequence_t<native_extent, 1, Is...>{}, dst, src);
	}

	template<typename T, std::size_t N, typename V, typename Op>
	DPM_FORCEINLINE T reduce(const V *data, V idt, Op op) noexcept
	{
		/* Reduce every element vertically, then horizontally within a single vector. */
		constexpr auto native_extent = sizeof(V) / sizeof(T);
		V res;
		for (std::size_t i = 0; i < N; i += native_extent)
		{
			auto v = data[i / native_extent];
			if (i + native_extent > N) v = maskblend<T>(v, idt, N - i);
			res = i ? op(res, v) : v;
		}
		return reduce<T>(res, op);
	}
}