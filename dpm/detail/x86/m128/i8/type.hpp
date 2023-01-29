/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../type_fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

namespace dpm
{
	namespace detail
	{
		/* When `n` is < 16, mix `16 - n` elements of `b` at the end of `a`. */
		DPM_FORCEINLINE __m128i maskblend_i8(std::size_t n, __m128i a, __m128i b) noexcept
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
			switch (const auto mask = static_cast<std::int8_t>(0xff); n)
			{
				case 15: vm = _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 14: vm = _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 13: vm = _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 12: vm = _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 11: vm = _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 10: vm = _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 9: vm = _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 8: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 7: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0);
					break;
				case 6: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0);
					break;
				case 5: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0);
					break;
				case 4: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0);
					break;
				case 3: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0);
					break;
				case 2: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0);
					break;
				case 1: vm = _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0);
					break;
				default: return a;
			}
			return _mm_or_si128(_mm_andnot_si128(vm, a), _mm_and_si128(vm, b));
#endif
		}
		/* When `n` is < 16, mix `16 - n` zeros at the end of `a`. */
		DPM_FORCEINLINE __m128i maskzero_i8(std::size_t n, __m128i v) noexcept
		{
			switch (const auto m = static_cast<std::int8_t>(0xff); n)
			{
				case 15: return _mm_and_si128(v, _mm_set_epi8(0, m, m, m, m, m, m, m, m, m, m, m, m, m, m, m));
				case 14: return _mm_and_si128(v, _mm_set_epi8(0, 0, m, m, m, m, m, m, m, m, m, m, m, m, m, m));
				case 13: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, m, m, m, m, m, m, m, m, m, m, m, m, m));
				case 12: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m, m, m));
				case 11: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m, m));
				case 10: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m, m));
				case 9: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m, m));
				case 8: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m, m));
				case 7: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m, m));
				case 6: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m, m));
				case 5: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m, m));
				case 4: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m, m));
				case 3: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m, m));
				case 2: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m, m));
				case 1: return _mm_and_si128(v, _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, m));
				default: return v;
			}
		}
		/* When `n` is < 8, mix `8 - n` ones at the end of `a`. */
		DPM_FORCEINLINE __m128i maskone_i8(std::size_t n, __m128i v) noexcept
		{
			switch (const auto m = static_cast<std::int8_t>(0xff); n)
			{
				case 15: return _mm_or_si128(v, _mm_set_epi8(m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 14: return _mm_or_si128(v, _mm_set_epi8(m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 13: return _mm_or_si128(v, _mm_set_epi8(m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 12: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 11: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 10: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 9: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0, 0));
				case 8: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0, 0));
				case 7: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0, 0));
				case 6: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0, 0));
				case 5: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0, 0));
				case 4: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0, 0));
				case 3: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0, 0));
				case 2: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0, 0));
				case 1: return _mm_or_si128(v, _mm_set_epi8(m, m, m, m, m, m, m, m, m, m, m, m, m, m, m, 0));
				default: return v;
			}
		}

#ifdef DPM_HAS_SSSE3
		template<std::size_t I, std::size_t... Is>
		inline void shuffle_i8(__m128i *, const __m128i *) noexcept requires (sizeof...(Is) < 15);
		template<std::size_t I>
		DPM_FORCEINLINE void shuffle_i8(__m128i *to, const __m128i *from) noexcept
		{
			*to = _mm_shuffle_epi8(from[I / 16], _mm_set1_epi8(I % 16));
		}
		template<std::size_t I0, std::size_t I1, std::size_t I2, std::size_t I3, std::size_t I4, std::size_t I5, std::size_t I6, std::size_t I7,
				std::size_t I8, std::size_t I9, std::size_t I10, std::size_t I11, std::size_t I12, std::size_t I13, std::size_t I14, std::size_t I15,
				std::size_t... Is>
		DPM_FORCEINLINE void shuffle_i8(__m128i *to, const __m128i *from) noexcept
		{
			constexpr auto P0 = I0 / 16, P1 = I1 / 16, P2 = I2 / 16, P3 = I3 / 16, P4 = I4 / 16, P5 = I5 / 16, P6 = I6 / 16, P7 = I7 / 16,
					P8 = I8 / 16, P9 = I9 / 16, P10 = I10 / 16, P11 = I11 / 16, P12 = I12 / 16, P13 = I13 / 16, P14 = I14 / 16, P15 = I15 / 16;
			if constexpr (P0 == P1 && P1 == P2 && P2 == P3 && P3 == P4 && P4 == P5 && P5 == P6 && P6 == P7 && P7 == P8 && P8 == P9 && P10 == P11 && P11 == P12 && P12 == P13 && P13 == P14 && P13 == P15)
				*to = _mm_shuffle_epi8(from[P0], _mm_set_epi8(I0 % 16, I1 % 16, I2 % 16, I3 % 16, I4 % 16, I5 % 16, I6 % 16, I7 % 16, I8 % 16, I9 % 16, I10 % 16, I11 % 16, I12 % 16, I13 % 16, I14 % 16, I15 % 16));
			else
				copy_positions<I0, I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11, I12, I13, I14, I15>(reinterpret_cast<alias_uint8_t *>(to), reinterpret_cast<const alias_uint8_t *>(from));
			if constexpr (sizeof...(Is) != 0) shuffle_i8<Is...>(to + 1, from);
		}
		template<std::size_t I, std::size_t... Is>
		DPM_FORCEINLINE void shuffle_i8(__m128i *to, const __m128i *from) noexcept requires (sizeof...(Is) < 15)
		{
			shuffle_i8<I, Is..., I>(to, from);
		}
#endif
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_type<simd_mask<I, detail::avec<N, Align>>> { using type = __m128i; };
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_size<simd_mask<I, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<I, N, 16>()> {};

		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd_mask<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd_mask<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
	}

	template<detail::integral_of_size<1> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
	class simd_mask<I, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max(Align, 16);

		using value_alias = detail::basic_mask<std::int8_t>;
		using storage_type = __m128i[data_size];

	public:
		using value_type = bool;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<I, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the SIMD mask object with a native SSE mask vector.
		 * @note This constructor is available for overload resolution only when the SIMD mask contains a single SSE vector. */
		constexpr simd_mask(__m128i native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128i)`. */
		constexpr simd_mask(const __m128i (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff)) : _mm_setzero_si128();
			std::fill_n(m_data, data_size, v);
		}
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && alignof(decltype(other)) >= alignment)
				std::copy_n(reinterpret_cast<const __m128i *>(ext::to_native_data(other).data()), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 16)
			{
				std::int8_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0, i8 = 0, i9 = 0, i10 = 0, i11 = 0, i12 = 0, i13 = 0, i14 = 0, i15 = 0;
				switch (size() - i)
				{
					default: i15 = detail::extend_bool<std::int8_t>(mem[i + 15]); [[fallthrough]];
					case 15: i14 = detail::extend_bool<std::int8_t>(mem[i + 14]); [[fallthrough]];
					case 14: i13 = detail::extend_bool<std::int8_t>(mem[i + 13]); [[fallthrough]];
					case 13: i12 = detail::extend_bool<std::int8_t>(mem[i + 12]); [[fallthrough]];
					case 12: i11 = detail::extend_bool<std::int8_t>(mem[i + 11]); [[fallthrough]];
					case 11: i10 = detail::extend_bool<std::int8_t>(mem[i + 10]); [[fallthrough]];
					case 10: i9 = detail::extend_bool<std::int8_t>(mem[i + 9]); [[fallthrough]];
					case 9: i8 = detail::extend_bool<std::int8_t>(mem[i + 8]); [[fallthrough]];
					case 8: i7 = detail::extend_bool<std::int8_t>(mem[i + 7]); [[fallthrough]];
					case 7: i6 = detail::extend_bool<std::int8_t>(mem[i + 6]); [[fallthrough]];
					case 6: i5 = detail::extend_bool<std::int8_t>(mem[i + 5]); [[fallthrough]];
					case 5: i4 = detail::extend_bool<std::int8_t>(mem[i + 4]); [[fallthrough]];
					case 4: i3 = detail::extend_bool<std::int8_t>(mem[i + 3]); [[fallthrough]];
					case 3: i2 = detail::extend_bool<std::int8_t>(mem[i + 2]); [[fallthrough]];
					case 2: i1 = detail::extend_bool<std::int8_t>(mem[i + 1]); [[fallthrough]];
					case 1: i0 = detail::extend_bool<std::int8_t>(mem[i]);
				}
				m_data[i / 16] = _mm_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
			}
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 16)
				switch (const auto bits = _mm_movemask_epi8(m_data[i / 16]); size() - i)
				{
					default: mem[i + 15] = bits & 0b1000'0000'0000'0000; [[fallthrough]];
					case 15: mem[i + 14] = bits & 0b0100'0000'0000'0000; [[fallthrough]];
					case 14: mem[i + 13] = bits & 0b0010'0000'0000'0000; [[fallthrough]];
					case 13: mem[i + 12] = bits & 0b0001'0000'0000'0000; [[fallthrough]];
					case 12: mem[i + 11] = bits & 0b0000'1000'0000'0000; [[fallthrough]];
					case 11: mem[i + 10] = bits & 0b0000'0100'0000'0000; [[fallthrough]];
					case 10: mem[i + 9] = bits & 0b0000'0010'0000'0000; [[fallthrough]];
					case 9: mem[i + 8] = bits & 0b0000'0001'0000'0000; [[fallthrough]];
					case 8: mem[i + 7] = bits & 0b0000'0000'1000'0000; [[fallthrough]];
					case 7: mem[i + 6] = bits & 0b0000'0000'0100'0000; [[fallthrough]];
					case 6: mem[i + 5] = bits & 0b0000'0000'0010'0000; [[fallthrough]];
					case 5: mem[i + 4] = bits & 0b0000'0000'0001'0000; [[fallthrough]];
					case 4: mem[i + 3] = bits & 0b0000'0000'0000'1000; [[fallthrough]];
					case 3: mem[i + 2] = bits & 0b0000'0000'0000'0100; [[fallthrough]];
					case 2: mem[i + 1] = bits & 0b0000'0000'0000'0010; [[fallthrough]];
					case 1: mem[i] = bits & 0b0000'0000'0000'0001;
				}
		}

		[[nodiscard]] DPM_FORCEINLINE reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data)[i];
		}
		[[nodiscard]] DPM_FORCEINLINE value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data)[i];
		}

		[[nodiscard]] DPM_FORCEINLINE simd_mask operator!() const noexcept
		{
			simd_mask result = {};
			const auto mask = _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_si128(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_and_si128(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_or_si128(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_si128(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_and_si128(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_or_si128(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_xor_si128(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept { return a & b; }
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept { return a | b; }

		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_cmpeq_epi8(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept { return !(a == b); }

	private:
		alignas(alignment) storage_type m_data;
	};

	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
	class const_where_expression<simd_mask<I, detail::avec<N, A>>, simd_mask<I, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		using value_type = bool;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, mask_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const mask_t &data) noexcept : m_mask(mask), m_data(const_cast<mask_t &>(data)) {}

		/** Copies selected elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_mask);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); i += 16)
				switch (const auto bits = _mm_movemask_epi8(_mm_and_si128(v_data[i / 16], v_mask[i / 16])); mask_t::size() - i)
				{
					default: mem[i + 15] = bits & 0b1000'0000'0000'0000; [[fallthrough]];
					case 15: mem[i + 14] = bits & 0b0100'0000'0000'0000; [[fallthrough]];
					case 14: mem[i + 13] = bits & 0b0010'0000'0000'0000; [[fallthrough]];
					case 13: mem[i + 12] = bits & 0b0001'0000'0000'0000; [[fallthrough]];
					case 12: mem[i + 11] = bits & 0b0000'1000'0000'0000; [[fallthrough]];
					case 11: mem[i + 10] = bits & 0b0000'0100'0000'0000; [[fallthrough]];
					case 10: mem[i + 9] = bits & 0b0000'0010'0000'0000; [[fallthrough]];
					case 9: mem[i + 8] = bits & 0b0000'0001'0000'0000; [[fallthrough]];
					case 8: mem[i + 7] = bits & 0b0000'0000'1000'0000; [[fallthrough]];
					case 7: mem[i + 6] = bits & 0b0000'0000'0100'0000; [[fallthrough]];
					case 6: mem[i + 5] = bits & 0b0000'0000'0010'0000; [[fallthrough]];
					case 5: mem[i + 4] = bits & 0b0000'0000'0001'0000; [[fallthrough]];
					case 4: mem[i + 3] = bits & 0b0000'0000'0000'1000; [[fallthrough]];
					case 3: mem[i + 2] = bits & 0b0000'0000'0000'0100; [[fallthrough]];
					case 2: mem[i + 1] = bits & 0b0000'0000'0000'0010; [[fallthrough]];
					case 1: mem[i] = bits & 0b0000'0000'0000'0001;
				}
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
	class where_expression<simd_mask<I, detail::avec<N, A>>, simd_mask<I, detail::avec<N, A>>> : public const_where_expression<simd_mask<I, detail::avec<N, A>>, simd_mask<I, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<I, detail::avec<N, A>>, simd_mask<I, detail::avec<N, A>>>;
		using value_type = typename base_expr::value_type;
		using mask_t = typename base_expr::mask_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<typename U>
		DPM_FORCEINLINE void operator=(U &&value) && noexcept requires std::is_convertible_v<U, value_type> { m_data = ext::blend(m_data, mask_t{std::forward<U>(value)}, m_mask); }

		template<typename U>
		DPM_FORCEINLINE void operator&=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data & mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator|=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data | mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator^=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data ^ mask_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_from(const bool *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_mask);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); i += 16)
			{
				std::int8_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0, i8 = 0, i9 = 0, i10 = 0, i11 = 0, i12 = 0, i13 = 0, i14 = 0, i15 = 0;
				switch (mask_t::size() - i)
				{
					default: i15 = detail::extend_bool<std::int8_t>(mem[i + 15]); [[fallthrough]];
					case 15: i14 = detail::extend_bool<std::int8_t>(mem[i + 14]); [[fallthrough]];
					case 14: i13 = detail::extend_bool<std::int8_t>(mem[i + 13]); [[fallthrough]];
					case 13: i12 = detail::extend_bool<std::int8_t>(mem[i + 12]); [[fallthrough]];
					case 12: i11 = detail::extend_bool<std::int8_t>(mem[i + 11]); [[fallthrough]];
					case 11: i10 = detail::extend_bool<std::int8_t>(mem[i + 10]); [[fallthrough]];
					case 10: i9 = detail::extend_bool<std::int8_t>(mem[i + 9]); [[fallthrough]];
					case 9: i8 = detail::extend_bool<std::int8_t>(mem[i + 8]); [[fallthrough]];
					case 8: i7 = detail::extend_bool<std::int8_t>(mem[i + 7]); [[fallthrough]];
					case 7: i6 = detail::extend_bool<std::int8_t>(mem[i + 6]); [[fallthrough]];
					case 6: i5 = detail::extend_bool<std::int8_t>(mem[i + 5]); [[fallthrough]];
					case 5: i4 = detail::extend_bool<std::int8_t>(mem[i + 4]); [[fallthrough]];
					case 4: i3 = detail::extend_bool<std::int8_t>(mem[i + 3]); [[fallthrough]];
					case 3: i2 = detail::extend_bool<std::int8_t>(mem[i + 2]); [[fallthrough]];
					case 2: i1 = detail::extend_bool<std::int8_t>(mem[i + 1]); [[fallthrough]];
					case 1: i0 = detail::extend_bool<std::int8_t>(mem[i]);
				}
				v_data[i / 16] = _mm_and_si128(_mm_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0), v_mask[i / 16]);
			}
		}
	};

	namespace detail
	{
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
		struct native_access<simd_mask<I, avec<N, A>>>
		{
			using mask_t = simd_mask<I, avec<N, A>>;

			[[nodiscard]] static std::span<__m128i, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128i, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd_mask<I, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd_mask<I, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSSE3
		/** Shuffles elements of mask \a x into a new mask according to the specified indices. */
		template<std::size_t... Is, detail::integral_of_size<1> I, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE simd_mask<I, detail::avec<M, A>> shuffle(const simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_any<I, N, A> && detail::x86_overload_128<I, M, A>
		{
			if constexpr (detail::is_sequential<0, Is...>::value && M == N)
				return simd_mask<I, detail::avec<M, A>>{x};
			else
			{
				simd_mask<I, detail::avec<M, A>> result = {};
				const auto src_data = reinterpret_cast<const __m128i *>(to_native_data(x).data());
				detail::shuffle_i8<Is...>(to_native_data(result).data(), src_data);
				return result;
			}
		}
#endif
	}

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool all_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

#ifdef DPM_HAS_SSE4_1
		if constexpr (ext::native_data_size_v<mask_t> == 1) return _mm_test_all_ones(detail::maskone_i8(mask_t::size(), mask_data[0]));
#endif
		auto result = _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 16)
		{
			const auto vm = detail::maskone_i8(mask_t::size() - i, mask_data[i / 16]);
			result = _mm_and_si128(result, vm);
		}
		return _mm_movemask_epi8(result) == 0xffff;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool any_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_si128();
		for (std::size_t i = 0; i < mask_t::size(); i += 16) result = _mm_or_si128(result, detail::maskone_i8(mask_t::size() - i, mask_data[i / 16]));

#ifdef DPM_HAS_SSE4_1
		return !_mm_testz_si128(result, result);
#else
		return _mm_movemask_epi8(result);
#endif
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool none_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_si128();
		for (std::size_t i = 0; i < mask_t::size(); i += 16) result = _mm_or_si128(result, detail::maskone_i8(mask_t::size() - i, mask_data[i / 16]));

#ifdef DPM_HAS_SSE4_1
		return _mm_testz_si128(result, result);
#else
		return !_mm_movemask_epi8(result);
#endif
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool some_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = _mm_setzero_si128(), all_mask = _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 16)
		{
			const auto vm = mask_data[i / 16];
			const auto vmz = detail::maskzero_i8(mask_t::size() - i, vm);
			const auto vmo = detail::maskone_i8(mask_t::size() - i, vm);

			all_mask = _mm_and_si128(all_mask, vmo);
			any_mask = _mm_or_si128(any_mask, vmz);
		}
#ifdef DPM_HAS_SSE4_1
		return !_mm_testz_si128(any_mask, any_mask) && !_mm_test_all_ones(all_mask);
#else
		return _mm_movemask_epi8(any_mask) && _mm_movemask_epi8(all_mask) != 0xffff;
#endif
	}

	/** Returns the number of `true` elements of \a mask. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t popcount(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < mask_t::size(); i += 16)
		{
			const auto vm = detail::maskzero_i8(mask_t::size() - i, mask_data[i / 16]);
			result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_epi8(vm)));
		}
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_first_set(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = static_cast<std::uint16_t>(_mm_movemask_epi8(mask_data[i]));
			if (bits) return std::countr_zero(bits) + i * 16;
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_last_set(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = ext::native_data_size_v<mask_t>, k; (k = i--) != 0;)
		{
			auto bits = static_cast<std::uint16_t>(_mm_movemask_epi8(mask_data[i]));
			switch (N - i * 16)
			{
				case 1: bits <<= 1; [[fallthrough]];
				case 2: bits <<= 1; [[fallthrough]];
				case 3: bits <<= 1; [[fallthrough]];
				case 4: bits <<= 1; [[fallthrough]];
				case 5: bits <<= 1; [[fallthrough]];
				case 6: bits <<= 1; [[fallthrough]];
				case 7: bits <<= 1; [[fallthrough]];
				case 8: bits <<= 1; [[fallthrough]];
				case 9: bits <<= 1; [[fallthrough]];
				case 10: bits <<= 1; [[fallthrough]];
				case 11: bits <<= 1; [[fallthrough]];
				case 12: bits <<= 1; [[fallthrough]];
				case 13: bits <<= 1; [[fallthrough]];
				case 14: bits <<= 1; [[fallthrough]];
				case 15: bits <<= 1; [[fallthrough]];
				default: break;
			}
			if (bits) return (k * 16 - 1) - std::countl_zero(bits);
		}
		DPM_UNREACHABLE();
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_type<simd<I, detail::avec<N, Align>>> { using type = __m128i; };
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_size<simd<I, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<I, N, 16>()> {};

		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
	}

	template<detail::integral_of_size<1> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
	class simd<I, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		constexpr static auto data_size = ext::native_data_size_v<simd>;
		constexpr static auto alignment = std::max(Align, 16);

		using value_alias = detail::simd_alias<I>;
		using storage_type = __m128i[data_size];

	public:
		using value_type = I;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<I, abi_type>;

		/** Returns width of the SIMD type. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the SIMD vector with a native SSE vector.
		 * @note This constructor is available for overload resolution only when the SIMD vector contains a single SSE vector. */
		constexpr simd(__m128i native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128i)`. */
		constexpr simd(const __m128i (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_epi8(static_cast<std::int8_t>(static_cast<I>(value)));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
		{
			detail::generate_n<data_size>(m_data, [&gen]<std::size_t J>(std::integral_constant<std::size_t, J>)
			{
				std::int8_t i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0, i8 = 0, i9 = 0, i10 = 0, i11 = 0, i12 = 0, i13 = 0, i14 = 0, i15 = 0;
				switch (constexpr auto value_idx = J * 16; size() - value_idx)
				{
					default: i15 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 15>()));
					case 15: i14 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 14>()));
					case 14: i13 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 13>()));
					case 13: i12 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 12>()));
					case 12: i11 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 11>()));
					case 11: i10 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 10>()));
					case 10: i9 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 9>()));
					case 9: i8 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 8>()));
					case 8: i7 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 7>()));
					case 7: i6 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 6>()));
					case 6: i5 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 5>()));
					case 5: i4 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 4>()));
					case 4: i3 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 3>()));
					case 3: i2 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 2>()));
					case 2: i1 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>()));
					case 1: i0 = static_cast<std::int8_t>(std::invoke(gen, std::integral_constant<std::size_t, value_idx>()));
				}
				return _mm_set_epi8(i15, i14, i13, i12, i11, i10, i9, i8, i7, i6, i5, i4, i3, i2, i1, i0);
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (constexpr auto other_alignment = alignof(decltype(other)); other_alignment >= alignment)
				copy_from(reinterpret_cast<const detail::simd_alias<U> *>(ext::to_native_data(other).data()), vector_aligned);
			else if constexpr (other_alignment != alignof(value_type))
				copy_from(reinterpret_cast<const detail::simd_alias<U> *>(ext::to_native_data(other).data()), overaligned<other_alignment>);
			else
				copy_from(reinterpret_cast<const detail::simd_alias<U> *>(ext::to_native_data(other).data()), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, 16> && detail::integral_of_size<std::remove_volatile_t<U>, 1>)
			{
				for (std::size_t i = 0; i < size(); i += 16)
					switch (size() - i)
					{
						default: m_data[i / 16] = reinterpret_cast<const __m128i *>(mem)[i / 16];
							break;
						case 15: operator[](i + 14) = static_cast<I>(mem[i + 14]); [[fallthrough]];
						case 14: operator[](i + 13) = static_cast<I>(mem[i + 13]); [[fallthrough]];
						case 13: operator[](i + 12) = static_cast<I>(mem[i + 12]); [[fallthrough]];
						case 12: operator[](i + 11) = static_cast<I>(mem[i + 11]); [[fallthrough]];
						case 11: operator[](i + 10) = static_cast<I>(mem[i + 10]); [[fallthrough]];
						case 10: operator[](i + 9) = static_cast<I>(mem[i + 9]); [[fallthrough]];
						case 9: operator[](i + 8) = static_cast<I>(mem[i + 8]); [[fallthrough]];
						case 8: operator[](i + 7) = static_cast<I>(mem[i + 7]); [[fallthrough]];
						case 7: operator[](i + 6) = static_cast<I>(mem[i + 6]); [[fallthrough]];
						case 6: operator[](i + 5) = static_cast<I>(mem[i + 5]); [[fallthrough]];
						case 5: operator[](i + 4) = static_cast<I>(mem[i + 4]); [[fallthrough]];
						case 4: operator[](i + 3) = static_cast<I>(mem[i + 3]); [[fallthrough]];
						case 3: operator[](i + 2) = static_cast<I>(mem[i + 2]); [[fallthrough]];
						case 2: operator[](i + 1) = static_cast<I>(mem[i + 1]); [[fallthrough]];
						case 1: operator[](i) = static_cast<I>(mem[i]);
					}
			}
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = static_cast<I>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::aligned_tag<Flags, 16> && detail::integral_of_size<std::remove_volatile_t<U>, 1>)
			{
				for (std::size_t i = 0; i < size(); i += 16)
					switch (size() - i)
					{
						default: reinterpret_cast<__m128i *>(mem)[i / 16] = m_data[i / 16];
							break;
						case 15: mem[i + 14] = static_cast<U>(operator[](i + 14)); [[fallthrough]];
						case 14: mem[i + 13] = static_cast<U>(operator[](i + 13)); [[fallthrough]];
						case 13: mem[i + 12] = static_cast<U>(operator[](i + 12)); [[fallthrough]];
						case 12: mem[i + 11] = static_cast<U>(operator[](i + 11)); [[fallthrough]];
						case 11: mem[i + 10] = static_cast<U>(operator[](i + 10)); [[fallthrough]];
						case 10: mem[i + 9] = static_cast<U>(operator[](i + 9)); [[fallthrough]];
						case 9: mem[i + 8] = static_cast<U>(operator[](i + 8)); [[fallthrough]];
						case 8: mem[i + 7] = static_cast<U>(operator[](i + 7)); [[fallthrough]];
						case 7: mem[i + 6] = static_cast<U>(operator[](i + 6)); [[fallthrough]];
						case 6: mem[i + 5] = static_cast<U>(operator[](i + 5)); [[fallthrough]];
						case 5: mem[i + 4] = static_cast<U>(operator[](i + 4)); [[fallthrough]];
						case 4: mem[i + 3] = static_cast<U>(operator[](i + 3)); [[fallthrough]];
						case 3: mem[i + 2] = static_cast<U>(operator[](i + 2)); [[fallthrough]];
						case 2: mem[i + 1] = static_cast<U>(operator[](i + 1)); [[fallthrough]];
						case 1: mem[i] = static_cast<U>(operator[](i));
					}
			}
			else
				for (std::size_t i = 0; i < size(); ++i) mem[i] = static_cast<U>(operator[](i));
		}

		[[nodiscard]] DPM_FORCEINLINE reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data)[i];
		}
		[[nodiscard]] DPM_FORCEINLINE value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data)[i];
		}

		DPM_FORCEINLINE simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		DPM_FORCEINLINE simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		DPM_FORCEINLINE simd &operator++() noexcept
		{
			const auto one = _mm_set1_epi8(1);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_add_epi8(m_data[i], one);
			return *this;
		}
		DPM_FORCEINLINE simd &operator--() noexcept
		{
			const auto one = _mm_set1_epi8(1);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_sub_epi8(m_data[i], one);
			return *this;
		}

		[[nodiscard]] DPM_FORCEINLINE mask_type operator!() const noexcept
		{
			mask_type result = {};
			for (std::size_t i = 0; i < size(); ++i) result[i] = !static_cast<bool>(operator[](i));
			return result;
		}
		[[nodiscard]] DPM_FORCEINLINE simd operator~() const noexcept
		{
			simd result = {};
			const auto mask = _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_si128(m_data[i], mask);
			return result;
		}
		[[nodiscard]] DPM_FORCEINLINE simd operator+() const noexcept { return *this; }
		[[nodiscard]] DPM_FORCEINLINE simd operator-() const noexcept requires std::is_signed_v<I>
		{
			simd result = {};
			const auto zero = _mm_setzero_si128();
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_epi8(zero, m_data[i]);
			return result;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd operator&(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_and_si128(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator|(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_or_si128(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator^(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_si128(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd &operator&=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_and_si128(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator|=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_or_si128(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator^=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_xor_si128(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_add_epi8(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_epi8(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd &operator+=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_add_epi8(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator-=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) b.m_data[i] = _mm_sub_epi8(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator==(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_epi8(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator!=(const simd &a, const simd &b) noexcept { return !(a == b); }

	private:
		alignas(alignment) storage_type m_data;
	};

	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
	class const_where_expression<simd_mask<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		using simd_t = simd<I, detail::avec<N, A>>;
		using value_type = I;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, simd_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const simd_t &data) noexcept : m_mask(mask), m_data(const_cast<simd_t &>(data)) {}

		[[nodiscard]] DPM_FORCEINLINE simd_t operator-() const && noexcept { return ext::blend(m_data, -m_data, m_mask); }
		[[nodiscard]] DPM_FORCEINLINE simd_t operator+() const && noexcept { return ext::blend(m_data, +m_data, m_mask); }

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (detail::integral_of_size<std::remove_volatile_t<U>, 1>)
			{
				const auto v_mask = ext::to_native_data(m_mask);
				const auto v_data = ext::to_native_data(m_data);
				for (std::size_t i = 0; i < mask_t::size(); i += 16)
				{
					const auto mi = detail::maskzero_i8(mask_t::size() - i, v_mask[i / 16]);
					_mm_maskmoveu_si128(v_data[i / 16], mi, reinterpret_cast<char *>(mem + i));
				}
			}
			else
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};

	namespace detail
	{
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
		struct native_access<simd<I, avec<N, A>>>
		{
			using simd_t = simd<I, avec<N, A>>;

			[[nodiscard]] static std::span<__m128i, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128i, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd<I, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd<I, detail::avec<N, A>>>::to_native_data(x);
		}

		/** Shuffles elements of vector \a x into a new vector according to the specified indices. */
		template<std::size_t... Is, detail::integral_of_size<1> I, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE simd<I, detail::avec<M, A>> shuffle(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_any<I, N, A> && detail::x86_overload_128<I, M, A>
		{
			if constexpr (detail::is_sequential<0, Is...>::value && M == N)
				return simd<I, detail::avec<M, A>>{x};
			else
			{
				simd<I, detail::avec<M, A>> result = {};
				const auto src_data = reinterpret_cast<const __m128i *>(to_native_data(x).data());
				detail::shuffle_i8<Is...>(to_native_data(result).data(), src_data);
				return result;
			}
		}
	}

#pragma region "simd reductions"
#ifdef DPM_HAS_SSSE3
	namespace detail
	{
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A, typename Op>
		DPM_FORCEINLINE __m128i reduce_lanes_i8(const simd<I, detail::avec<N, A>> &x, __m128i idt, Op op) noexcept
		{
			auto res = _mm_undefined_si128();
			for (std::size_t i = 0; i < x.size(); i += 16)
			{
				if (const auto v = maskblend_i8(x.size() - i, ext::to_native_data(x)[i / 16], idt); i != 0)
					res = op(res, v);
				else
					res = v;
			}
			return res;
		}
		template<detail::integral_of_size<1> I, std::size_t N, std::size_t A, typename Op>
		DPM_FORCEINLINE I reduce_i8(const simd<I, detail::avec<N, A>> &x, __m128i idt, Op op) noexcept
		{
			const auto a = reduce_lanes_i8(x, idt, op);
			auto b = _mm_shuffle_epi8(x, _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 8));
			auto c = op(a, b);
			b = _mm_shuffle_epi8(c, _mm_set_epi8(15, 14, 13, 12, 15, 14, 13, 12, 7, 6, 5, 4, 7, 6, 5, 4));
			c = op(c, b);
			b = _mm_shuffle_epi8(c, _mm_set_epi8(15, 14, 15, 14, 11, 10, 11, 10, 7, 6, 7, 6, 3, 2, 3, 2));
			c = op(c, b);
			b = _mm_shuffle_epi8(c, _mm_set1_epi8(0));
			return static_cast<I>(_mm_cvtsi128_si32(op(c, b)) & 0xff);
		}
	}

	/** Horizontally reduced elements of \a x using operation `Op`. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A, typename Op = std::plus<>>
	[[nodiscard]] DPM_FORCEINLINE I reduce(const simd<I, detail::avec<N, A>> &x, Op op = {}) noexcept requires detail::x86_overload_128<I, N, A>
	{
		if constexpr (std::same_as<Op, std::plus<>> || std::same_as<Op, std::plus<I>>)
			return detail::reduce_i8(x, _mm_setzero_si128(), [](auto a, auto b) { return _mm_add_epi8(a, b); });
		else
			return detail::reduce_impl<simd<I, detail::avec<N, A>>::size()>(x, op);
	}

#ifdef DPM_HAS_SSE4_1
	/** Calculates horizontal minimum of elements of \a x. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE I hmin(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
	{
		const auto max = std::numeric_limits<I>::max();
		return detail::reduce_i8(x, _mm_set1_epi8(max), [](auto a, auto b) { return _mm_min_epi8(a, b); });
	}
	/** Calculates horizontal maximum of elements of \a x. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE I hmax(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
	{
		const auto min = std::numeric_limits<I>::min();
		return detail::reduce_i8(x, _mm_set1_epi8(min), [](auto a, auto b) { return _mm_max_epi8(a, b); });
	}
#endif
#endif
#pragma endregion

#pragma region "simd algorithms"
#ifdef DPM_HAS_SSE4_1
/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<I, detail::avec<N, A>> min(
			const simd<I, detail::avec<N, A>> &a,
			const simd<I, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		simd<I, detail::avec<N, A>> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_epi8(a_data[i], b_data[i]);
		return result;
	}
/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<I, detail::avec<N, A>> max(
			const simd<I, detail::avec<N, A>> &a,
			const simd<I, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		simd<I, detail::avec<N, A>> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_max_epi8(a_data[i], b_data[i]);
		return result;
	}

/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::pair<simd<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>> minmax(
			const simd<I, detail::avec<N, A>> &a,
			const simd<I, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		std::pair<simd<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>> result = {};
		auto min_data = ext::to_native_data(result.first), max_data = ext::to_native_data(result.second);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < data_size; ++i)
		{
			min_data[i] = _mm_min_epi8(a_data[i], b_data[i]);
			max_data[i] = _mm_max_epi8(a_data[i], b_data[i]);
		}
		return result;
	}

/** Clamps elements of \a x between corresponding elements of \a ming and \a max. */
	template<detail::integral_of_size<1> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<I, detail::avec<N, A>> clamp(
			const simd<I, detail::avec<N, A>> &x,
			const simd<I, detail::avec<N, A>> &min,
			const simd<I, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		simd<I, detail::avec<N, A>> result = {};
		auto result_data = ext::to_native_data(result);
		const auto min_data = ext::to_native_data(min);
		const auto max_data = ext::to_native_data(max);
		const auto x_data = ext::to_native_data(x);

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_epi8(_mm_max_epi8(x_data[i], min_data[i]), max_data[i]);
		return result;
	}
#endif
#pragma endregion
}

#endif