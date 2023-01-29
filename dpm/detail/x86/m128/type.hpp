/*
 * Created by switchblade on 2023-01-07.
 */

#pragma once

#include "i8/type.hpp"
#include "i16/type.hpp"
#include "i32/type.hpp"
#include "i64/type.hpp"

#include "f32/type.hpp"
#include "f64/type.hpp"

#include "../type_fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "utility.hpp"
#include "bitwise.hpp"
#include "cmp.hpp"
#include "cvt.hpp"

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
		struct native_data_type<detail::x86_mask<T, N, Align>> { using type = __m128; };
		template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
		struct native_data_size<detail::x86_mask<T, N, Align>> : std::integral_constant<std::size_t, detail::align_data<T, N, 16>()> {};
	}

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_128<T, N, Align>
	class simd_mask<T, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max<std::size_t>(Align, 16);

		using storage_type = std::array<__m128, data_size>;
		using value_alias = detail::sized_mask<sizeof(T)>;

	public:
		using value_type = bool;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<T, abi_type>;

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
		constexpr simd_mask(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }

#ifdef DPM_HAS_SSE2
		/** @copydoc simd_mask */
		constexpr simd_mask(__m128i native) noexcept requires (data_size == 1) : simd_mask(std::bit_cast<__m128>(native)) {}
		/** @copydoc simd_mask */
		constexpr simd_mask(__m128d native) noexcept requires (data_size == 1) : simd_mask(std::bit_cast<__m128>(native)) {}
#endif

		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128i)`. */
		constexpr simd_mask(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

#ifdef DPM_HAS_SSE2
		/** @copydoc simd_mask */
		constexpr simd_mask(const __m128i (&native)[data_size]) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = std::bit_cast<__m128>(native[i]);
		}
		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128i)`. */
		constexpr simd_mask(const __m128d (&native)[data_size]) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = std::bit_cast<__m128>(native[i]);
		}
#endif

		/** Initializes the underlying elements with \a value. */
		DPM_FORCEINLINE simd_mask(value_type value) noexcept { m_data.fill(value ? detail::setones<__m128>() : detail::setzero<__m128>()); }
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_FORCEINLINE simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && alignof(decltype(other)) >= alignment)
				std::copy_n(reinterpret_cast<const __m128 *>(ext::to_native_data(other).data()), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 16 / sizeof(T)) detail::mask_copy<T>(mem, m_data[i], size() - i);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 16 / sizeof(T)) detail::mask_copy<T>(m_data[i], mem, size() - i);
		}

		[[nodiscard]] DPM_FORCEINLINE reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data.data())[i];
		}
		[[nodiscard]] DPM_FORCEINLINE value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data.data())[i];
		}

	private:
		alignas(alignment) storage_type m_data;
	};
}

#endif