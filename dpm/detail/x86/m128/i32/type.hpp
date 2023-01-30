/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../type_fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "../f32/utility.hpp"

namespace dpm
{
	namespace detail
	{
		/* When `n` is < 4, mix `4 - n` elements of `b` at the end of `a`. */
		DPM_FORCEINLINE __m128i maskblend_i32(std::size_t n, __m128i a, __m128i b) noexcept
		{
#ifdef DPM_HAS_SSE4_1
			switch (n)
			{
				case 3: return _mm_blend_epi16(a, b, 0b1100'0000);
				case 2: return _mm_blend_epi16(a, b, 0b1111'0000);
				case 1: return _mm_blend_epi16(a, b, 0b1111'1100);
				default: return a;
			}
#else
			auto vm = _mm_undefined_si128();
			switch (const auto mask = static_cast<std::int32_t>(0xffff'ffff); n)
			{
				case 3: vm = _mm_set_epi32(mask, 0, 0, 0);
					break;
				case 2: vm = _mm_set_epi32(mask, mask, 0, 0);
					break;
				case 1: vm = _mm_set_epi32(mask, mask, mask, 0);
					break;
				default: return a;
			}
			return _mm_or_si128(_mm_andnot_si128(vm, a), _mm_and_si128(vm, b));
#endif
		}
		/* When `n` is < 4, mix `4 - n` zeros at the end of `a`. */
		DPM_FORCEINLINE __m128i maskzero_i32(std::size_t n, __m128i v) noexcept
		{
			switch (const auto mask = static_cast<std::int32_t>(0xffff'ffff); n)
			{
				case 3: return _mm_and_si128(v, _mm_set_epi32(0, mask, mask, mask));
				case 2: return _mm_and_si128(v, _mm_set_epi32(0, 0, mask, mask));
				case 1: return _mm_and_si128(v, _mm_set_epi32(0, 0, 0, mask));
				default: return v;
			}
		}
		/* When `n` is < 4, mix `4 - n` ones at the end of `a`. */
		DPM_FORCEINLINE __m128i maskone_i32(std::size_t n, __m128i v) noexcept
		{
			switch (const auto mask = static_cast<std::int32_t>(0xffff'ffff); n)
			{
				case 3: return _mm_or_si128(v, _mm_set_epi32(mask, 0, 0, 0));
				case 2: return _mm_or_si128(v, _mm_set_epi32(mask, mask, 0, 0));
				case 1: return _mm_or_si128(v, _mm_set_epi32(mask, mask, mask, 0));
				default: return v;
			}
		}

		template<std::size_t I>
		DPM_FORCEINLINE void shuffle_i32(__m128i *to, const __m128i *from) noexcept
		{
			const auto v = std::bit_cast<__m128>(from[I / 4]);
			*to = _mm_shuffle_ps(v, v, _MM_SHUFFLE(I % 4, I % 4, I % 4, I % 4));
		}
		template<std::size_t I0, std::size_t I1>
		DPM_FORCEINLINE void shuffle_i32(__m128i *to, const __m128i *from) noexcept
		{
			constexpr auto P0 = I0 / 4, P1 = I1 / 4;
			if constexpr (P0 != P1)
				copy_elements<I0, I1>(reinterpret_cast<alias_uint32_t *>(to), reinterpret_cast<const alias_uint32_t *>(from));
			else
			{
				const auto v = std::bit_cast<__m128>(from[P0]);
				*to = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 2, I1 % 4, I0 % 4));
			}
		}
		template<std::size_t I0, std::size_t I1, std::size_t I2>
		DPM_FORCEINLINE void shuffle_i32(__m128i *to, const __m128i *from) noexcept
		{
			constexpr auto P0 = I0 / 4, P1 = I1 / 4, P2 = I2 / 4;
			if constexpr (P0 != P1)
				copy_elements<I0, I1, I2>(reinterpret_cast<alias_uint32_t *>(to), reinterpret_cast<const alias_uint32_t *>(from));
			else
			{
				const auto v0 = std::bit_cast<__m128>(from[P0]);
				const auto v2 = std::bit_cast<__m128>(from[P2]);
				*to = std::bit_cast<__m128i>(_mm_shuffle_ps(v0, v2, _MM_SHUFFLE(3, I2 % 4, I1 % 4, I0 % 4)));
			}
		}
		template<std::size_t I0, std::size_t I1, std::size_t I2, std::size_t I3, std::size_t... Is>
		DPM_FORCEINLINE void shuffle_i32(__m128i *to, const __m128i *from) noexcept
		{
			constexpr auto P0 = I0 / 4, P1 = I1 / 4, P2 = I2 / 4, P3 = I3 / 4;
			if constexpr (P0 != P1 || P2 != P3)
				copy_elements<I0, I1, I2, I3>(reinterpret_cast<alias_uint32_t *>(to), reinterpret_cast<const alias_uint32_t *>(from));
			else
			{
				const auto v0 = std::bit_cast<__m128>(from[P0]);
				const auto v2 = std::bit_cast<__m128>(from[P2]);
				*to = std::bit_cast<__m128i>(_mm_shuffle_ps(v0, v2, _MM_SHUFFLE(I3 % 4, I2 % 4, I1 % 4, I0 % 4)));
			}
			if constexpr (sizeof...(Is) != 0) shuffle_i32<Is...>(to + 1, from);
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_type<simd_mask<I, detail::avec<N, Align>>> { using type = __m128i; };
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_size<simd_mask<I, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<I, N, 16>()> {};

		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd_mask<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd_mask<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
	}

	namespace detail
	{
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
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
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd_mask<I, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd_mask<I, detail::avec<N, A>>>::to_native_data(x);
		}

		/** Shuffles elements of mask \a x into a new mask according to the specified indices. */
		template<std::size_t... Is, detail::integral_of_size<4> I, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE simd_mask<I, detail::avec<M, A>> shuffle(const simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_any<I, N, A> && detail::x86_overload_128<I, M, A>
		{
			if constexpr (detail::is_sequential<0, Is...>::value && M == N)
				return simd_mask<I, detail::avec<M, A>>{x};
			else
			{
				simd_mask<I, detail::avec<M, A>> result = {};
				const auto src_data = reinterpret_cast<const __m128i *>(to_native_data(x).data());
				detail::shuffle_i32<Is...>(to_native_data(result).data(), src_data);
				return result;
			}
		}
	}

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool all_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

#ifdef DPM_HAS_SSE4_1
		if constexpr (ext::native_data_size_v<mask_t> == 1) return _mm_test_all_ones(detail::maskone_i32(mask_t::size(), mask_data[0]));
#endif
		auto result = _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = detail::maskone_i32(mask_t::size() - i, mask_data[i / 4]);
			result = _mm_and_si128(result, vm);
		}
		return _mm_movemask_epi8(result) == 0xffff;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool any_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_si128();
		for (std::size_t i = 0; i < mask_t::size(); i += 4) result = _mm_or_si128(result, detail::maskone_i32(mask_t::size() - i, mask_data[i / 4]));

#ifdef DPM_HAS_SSE4_1
		return !_mm_testz_si128(result, result);
#else
		return _mm_movemask_epi8(result);
#endif
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool none_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = _mm_setzero_si128();
		for (std::size_t i = 0; i < mask_t::size(); i += 4) result = _mm_or_si128(result, detail::maskone_i32(mask_t::size() - i, mask_data[i / 4]));

#ifdef DPM_HAS_SSE4_1
		return _mm_testz_si128(result, result);
#else
		return !_mm_movemask_epi8(result);
#endif
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool some_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = _mm_setzero_si128(), all_mask = _mm_set1_epi32(static_cast<std::int32_t>(0xffff'ffff));
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = mask_data[i / 4];
			const auto vmz = detail::maskzero_i32(mask_t::size() - i, vm);
			const auto vmo = detail::maskone_i32(mask_t::size() - i, vm);

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
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t popcount(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < mask_t::size(); i += 4)
		{
			const auto vm = detail::maskzero_i32(mask_t::size() - i, mask_data[i / 4]);
			result += std::popcount(_mm_movemask_ps(std::bit_cast<__m128>(vm)));
		}
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_first_set(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(std::bit_cast<__m128>(mask_data[i])));
			if (bits) return std::countr_zero(bits) + i * 4;
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_last_set(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_128<I, N, A>
	{
		using mask_t = simd_mask<I, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = ext::native_data_size_v<mask_t>, k; (k = i--) != 0;)
		{
			auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(std::bit_cast<__m128>(mask_data[i])));
			switch (mask_t::size() - i * 4)
			{
				case 1: bits <<= 1; [[fallthrough]];
				case 2: bits <<= 1; [[fallthrough]];
				case 3: bits <<= 1; [[fallthrough]];
				default: bits <<= 4;
			}
			if (bits) return (k * 4 - 1) - std::countl_zero(bits);
		}
		DPM_UNREACHABLE();
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_type<simd<I, detail::avec<N, Align>>> { using type = __m128i; };
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
		struct native_data_size<simd<I, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<I, N, 16>()> {};

		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_128<I, N, A>;
	}

	template<detail::integral_of_size<4> I, std::size_t N, std::size_t Align> requires detail::x86_overload_128<I, N, Align>
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
			const auto vec = _mm_set1_epi32(static_cast<std::int32_t>(static_cast<I>(value)));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
		{
			detail::generate_n<data_size>(m_data, [&gen]<std::size_t J>(std::integral_constant<std::size_t, J>)
			{
				I i0 = 0, i1 = 0, i2 = 0, i3 = 0;
				switch (constexpr auto value_idx = J * 4; size() - value_idx)
				{
					default: i3 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 3>());
					case 3: i2 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 2>());
					case 2: i1 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>());
					case 1: i0 = std::invoke(gen, std::integral_constant<std::size_t, value_idx>());
				}
				return _mm_set_epi32(static_cast<std::int32_t>(i3), static_cast<std::int32_t>(i2), static_cast<std::int32_t>(i1), static_cast<std::int32_t>(i0));
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
			if constexpr (detail::aligned_tag<Flags, 16> && detail::integral_of_size<std::remove_volatile_t<U>, 4>)
			{
				for (std::size_t i = 0; i < size(); i += 4)
					switch (size() - i)
					{
						default: m_data[i / 4] = reinterpret_cast<const __m128i *>(mem)[i / 4];
							break;
						case 3: operator[](i + 2) = static_cast<I>(mem[i + 2]); [[fallthrough]];
						case 2: operator[](i + 1) = static_cast<I>(mem[i + 1]); [[fallthrough]];
						case 1: operator[](i) = static_cast<I>(mem[i]);
					}
			}
			else if constexpr (detail::aligned_tag<Flags, 16> && std::same_as<std::remove_volatile_t<U>, float>)
			{
				for (std::size_t i = 0; i < size(); i += 4)
					switch (size() - i)
					{
						default:
						{
							if constexpr (!std::is_signed_v<I>)
								m_data[i / 4] = detail::cvt_f32_u32(reinterpret_cast<const __m128 *>(mem)[i / 4]);
							else
								m_data[i / 4] = _mm_cvtps_epi32(reinterpret_cast<const __m128 *>(mem)[i / 4]);
							break;
						}
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
			if constexpr (detail::aligned_tag<Flags, 16> && detail::integral_of_size<std::remove_volatile_t<U>, 4>)
			{
				for (std::size_t i = 0; i < size(); i += 4)
					switch (size() - i)
					{
						default: reinterpret_cast<__m128i *>(mem)[i / 4] = m_data[i / 4];
							break;
						case 3: mem[i + 2] = static_cast<U>(operator[](i + 2)); [[fallthrough]];
						case 2: mem[i + 1] = static_cast<U>(operator[](i + 1)); [[fallthrough]];
						case 1: mem[i] = static_cast<U>(operator[](i));
					}
			}
			else if constexpr (detail::aligned_tag<Flags, 16> && std::same_as<std::remove_volatile_t<U>, float>)
			{
				for (std::size_t i = 0; i < size(); i += 4)
					switch (size() - i)
					{
						default:
						{
							if constexpr (!std::is_signed_v<I>)
								reinterpret_cast<__m128 *>(mem)[i / 4] = detail::cvt_u32_f32(m_data[i / 4]);
							else
								reinterpret_cast<__m128 *>(mem)[i / 4] = _mm_cvtepi32_ps(m_data[i / 4]);
						}
							break;
						case 3: mem[i + 2] = static_cast<float>(operator[](i + 2)); [[fallthrough]];
						case 2: mem[i + 1] = static_cast<float>(operator[](i + 1)); [[fallthrough]];
						case 1: mem[i] = static_cast<float>(operator[](i));
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
			const auto one = _mm_set1_epi32(1);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_add_epi32(m_data[i], one);
			return *this;
		}
		DPM_FORCEINLINE simd &operator--() noexcept
		{
			const auto one = _mm_set1_epi32(1);
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_sub_epi32(m_data[i], one);
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
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_epi32(zero, m_data[i]);
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
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_add_epi32(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_epi32(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd &operator+=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_add_epi32(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator-=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) b.m_data[i] = _mm_sub_epi32(a.m_data[i], b.m_data[i]);
			return a;
		}

#ifdef DPM_HAS_AVX2
		[[nodiscard]] friend DPM_FORCEINLINE simd operator<<(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sllv_epi32(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd operator>>(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_srlv_epi32(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend DPM_FORCEINLINE simd &operator<<=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_sllv_epi32(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd &operator>>=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_srlv_epi32(a.m_data[i], b.m_data[i]);
			return a;
		}
#endif

#ifdef DPM_HAS_SSE4_1
		[[nodiscard]] friend DPM_FORCEINLINE simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_mullo_epi32(a.m_data[i], b.m_data[i]);
			return result;
		}
		friend DPM_FORCEINLINE simd &operator*=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_mullo_epi32(a.m_data[i], b.m_data[i]);
			return a;
		}
#endif

		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator==(const simd &a, const simd &b) noexcept
		{
			storage_type mask_data = {};
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_epi32(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_FORCEINLINE mask_type operator!=(const simd &a, const simd &b) noexcept { return !(a == b); }

	private:
		alignas(alignment) storage_type m_data;
	};

	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
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
			if constexpr (sizeof(U) == 4)
			{
				const auto v_mask = ext::to_native_data(m_mask);
				const auto v_data = ext::to_native_data(m_data);
				for (std::size_t i = 0; i < mask_t::size(); i += 4)
				{
					const auto mi = detail::maskzero_i32(mask_t::size() - i, v_mask[i / 4]);
					auto v = v_data[i / 4];
					if constexpr (std::same_as<std::remove_volatile_t<U>, float>)
					{
						if constexpr (!std::is_signed_v<I>)
							v = std::bit_cast<__m128i>(detail::cvt_u32_f32(v));
						else
							v = std::bit_cast<__m128i>(detail::cvt_u32_f32(v));
					}
#ifdef DPM_HAS_AVX
					if constexpr (detail::aligned_tag<Flags, 16>)
						_mm_maskstore_ps(mem + i, mi, std::bit_cast<__m128>(v));
					else
#endif
						_mm_maskmoveu_si128(v, mi, reinterpret_cast<char *>(mem + i));
				}
			}
			else
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) mem[i] = static_cast<U>(m_data[i]);
		}

	protected:
		mask_t m_mask;
		simd_t &m_data;
	};
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
	class where_expression<simd_mask<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>> : public const_where_expression<simd_mask<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>>
	{
		using base_expr = const_where_expression<simd_mask<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>>;
		using value_type = typename base_expr::value_type;
		using simd_t = typename base_expr::simd_t;
		using mask_t = typename base_expr::mask_t;

		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<typename U>
		DPM_FORCEINLINE void operator=(U &&value) && noexcept requires std::is_convertible_v<U, value_type> { m_data = ext::blend(m_data, simd_t{std::forward<U>(value)}, m_mask); }

		DPM_FORCEINLINE void operator++() && noexcept
		{
			const auto old_data = m_data;
			const auto new_data = ++old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		DPM_FORCEINLINE void operator--() && noexcept
		{
			const auto old_data = m_data;
			const auto new_data = --old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		DPM_FORCEINLINE void operator++(int) && noexcept
		{
			const auto old_data = m_data++;
			m_data = ext::blend(old_data, m_data, m_mask);
		}
		DPM_FORCEINLINE void operator--(int) && noexcept
		{
			const auto old_data = m_data--;
			m_data = ext::blend(old_data, m_data, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator+=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data + simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator-=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data - simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		DPM_FORCEINLINE void operator*=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data * simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		DPM_FORCEINLINE void operator/=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			const auto new_data = m_data / simd_t{std::forward<U>(value)};
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename U, typename Flags>
		DPM_FORCEINLINE void copy_from(const U *mem, Flags) && noexcept requires is_simd_flag_type_v<Flags>
		{
#ifdef DPM_HAS_AVX
			if constexpr (detail::aligned_tag<Flags, 16> && sizeof(U) == 4)
			{
				const auto v_mask = ext::to_native_data(m_mask);
				const auto v_data = ext::to_native_data(m_data);
				for (std::size_t i = 0; i < mask_t::size(); i += 4)
				{
					const auto mi = detail::maskzero_i32(mask_t::size() - i, v_mask[i / 4]);
					auto v = std::bit_cast<__m128i>(_mm_maskload_ps(mem + i, mi));
					if constexpr (std::same_as<std::remove_volatile_t<U>, float>)
					{
						if constexpr (!std::is_signed_v<I>)
							v = detail::cvt_f32_u32(std::bit_cast<__m128>(v));
						else
							v = _mm_cvtps_epi32(std::bit_cast<__m128>(v));
					}
					v_data[i / 4] = v;
				}
			}
			else
#endif
				for (std::size_t i = 0; i < mask_t::size(); ++i) if (m_mask[i]) m_data[i] = static_cast<I>(mem[i]);
		}
	};

	namespace detail
	{
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A> requires detail::x86_overload_128<I, N, A>
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
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd<I, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
		{
			return detail::native_access<simd<I, detail::avec<N, A>>>::to_native_data(x);
		}

		/** Shuffles elements of vector \a x into a new vector according to the specified indices. */
		template<std::size_t... Is, detail::integral_of_size<4> I, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE simd<I, detail::avec<M, A>> shuffle(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_any<I, N, A> && detail::x86_overload_128<I, M, A>
		{
			if constexpr (detail::is_sequential<0, Is...>::value && M == N)
				return simd<I, detail::avec<M, A>>{x};
			else
			{
				simd<I, detail::avec<M, A>> result = {};
				const auto src_data = reinterpret_cast<const __m128i *>(to_native_data(x).data());
				detail::shuffle_i32<Is...>(to_native_data(result).data(), src_data);
				return result;
			}
		}
	}

#pragma region "simd reductions"
	namespace detail
	{
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A, typename Op>
		DPM_FORCEINLINE __m128i reduce_lanes_i32(const simd<I, detail::avec<N, A>> &x, __m128i idt, Op op) noexcept
		{
			auto res = _mm_undefined_si128();
			for (std::size_t i = 0; i < x.size(); i += 4)
			{
				if (const auto v = maskblend_i32(x.size() - i, ext::to_native_data(x)[i / 4], idt); i != 0)
					res = op(res, v);
				else
					res = v;
			}
			return res;
		}
		template<detail::integral_of_size<4> I, std::size_t N, std::size_t A, typename Op>
		DPM_FORCEINLINE I reduce_i32(const simd<I, detail::avec<N, A>> &x, __m128i idt, Op op) noexcept
		{
			const auto a = reduce_lanes_i32(x, idt, op);
			const auto b = _mm_shuffle_ps(std::bit_cast<__m128>(x), std::bit_cast<__m128>(x), _MM_SHUFFLE(3, 3, 1, 1));
			const auto c = op(a, b);
			const auto d = _mm_movehl_ps(std::bit_cast<__m128>(b), std::bit_cast<__m128>(c));
			return static_cast<I>(_mm_cvtsi128_si32(op(c, std::bit_cast<__m128>(d))));
		}
	}

	/** Horizontally reduced elements of \a x using operation `Op`. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A, typename Op = std::plus<>>
	[[nodiscard]] DPM_FORCEINLINE I reduce(const simd<I, detail::avec<N, A>> &x, Op op = {}) noexcept requires detail::x86_overload_128<I, N, A>
	{
		if constexpr (std::same_as<Op, std::plus<>> || std::same_as<Op, std::plus<I>>)
			return detail::reduce_i32(x, _mm_setzero_si128(), [](auto a, auto b) { return _mm_add_epi32(a, b); });
#ifdef DPM_HAS_SSE4_1
		else if constexpr (std::same_as<Op, std::multiplies<>> || std::same_as<Op, std::multiplies<I>>)
			return detail::reduce_i32(x, _mm_set1_epi32(1), [](auto a, auto b) { return _mm_mullo_epi32(a, b); });
#endif
		else
			return detail::reduce_impl<simd<I, detail::avec<N, A>>::size()>(x, op);
	}

#ifdef DPM_HAS_SSE4_1
	/** Calculates horizontal minimum of elements of \a x. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE I hmin(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
	{
		const auto max = std::numeric_limits<I>::max();
		return detail::reduce_i32(x, _mm_set1_epi32(max), [](auto a, auto b) { return _mm_min_epi32(a, b); });
	}
	/** Calculates horizontal maximum of elements of \a x. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE I hmax(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_128<I, N, A>
	{
		const auto min = std::numeric_limits<I>::min();
		return detail::reduce_i32(x, _mm_set1_epi32(min), [](auto a, auto b) { return _mm_max_epi32(a, b); });
	}
#endif
#pragma endregion

#pragma region "simd algorithms"
#ifdef DPM_HAS_SSE4_1
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
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

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_epi32(a_data[i], b_data[i]);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
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

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_max_epi32(a_data[i], b_data[i]);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
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
			min_data[i] = _mm_min_epi32(a_data[i], b_data[i]);
			max_data[i] = _mm_max_epi32(a_data[i], b_data[i]);
		}
		return result;
	}

	/** Clamps elements of \a x between corresponding elements of \a ming and \a max. */
	template<detail::integral_of_size<4> I, std::size_t N, std::size_t A>
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

		for (std::size_t i = 0; i < data_size; ++i) result_data[i] = _mm_min_epi32(_mm_max_epi32(x_data[i], min_data[i]), max_data[i]);
		return result;
	}
#endif
#pragma endregion
}

#endif