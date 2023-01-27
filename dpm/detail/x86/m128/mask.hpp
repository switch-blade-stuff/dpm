/*
 * Created by switch_blade on 2023-01-24.
 */

#pragma once

#include "utility.hpp"
#include "bitwise.hpp"
#include "cmp.hpp"
#include "cvt.hpp"

namespace dpm
{
	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, std::size_t N, std::size_t Align> requires detail::overload_128<T, N, Align>
		struct native_data_type<simd_mask<T, detail::avec<N, Align>>> { using type = __m128; };
		template<typename T, std::size_t N, std::size_t Align> requires detail::overload_128<T, N, Align>
		struct native_data_size<simd_mask<T, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<T, N, 16>()> {};
	}

	template<typename T, std::size_t N, std::size_t Align> requires detail::overload_128<T, N, Align>
	class simd_mask<T, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max<std::size_t>(Align, 16);
		constexpr static auto vector_extent = 16 / sizeof(T);

		using value_alias = detail::sized_mask<sizeof(T)>;
		using storage_type = __m128[data_size];

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
		/** @copydoc simd_mask */
		constexpr simd_mask(__m128i native) noexcept requires (data_size == 1) : simd_mask(std::bit_cast<__m128>(native)) {}
		/** @copydoc simd_mask */
		constexpr simd_mask(__m128d native) noexcept requires (data_size == 1) : simd_mask(std::bit_cast<__m128>(native)) {}

		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128i)`. */
		constexpr simd_mask(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }
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

		/** Initializes the underlying elements with \a value. */
		simd_mask(value_type value) noexcept { std::fill_n(m_data, data_size, value ? detail::setones<__m128>() : detail::setzero<__m128>()); }
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && alignof(decltype(other)) >= alignment)
				std::copy_n(reinterpret_cast<const __m128 *>(ext::to_native_data(other).data()), data_size, m_data);
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
			for (std::size_t i = 0; i < size(); i += vector_extent) detail::mask_copy<T>(mem, m_data[i], size() - i);
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += vector_extent) detail::mask_copy<T>(m_data[i], mem, size() - i);
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
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = detail::bit_not(m_data[i]);
			return result;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = detail::bit_and(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = detail::bit_xor(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = detail::bit_or(a.m_data[i], b.m_data[i]);
			return result;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept { return a & b; }
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept { return a | b; }

		friend DPM_FORCEINLINE simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = detail::bit_and(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = detail::bit_xor(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend DPM_FORCEINLINE simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = detail::bit_or(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = detail::mask_eq(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend DPM_FORCEINLINE simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
		{
			return !(a == b);
		}

	private:
		alignas(alignment) storage_type m_data;
	};

	namespace detail
	{
		template<typename T, std::size_t N, std::size_t A> requires detail::overload_128<T, N, A>
		struct native_access<simd_mask<T, avec<N, A>>>
		{
			using mask_t = simd_mask<T, avec<N, A>>;

			static std::span<__m128, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			static std::span<const __m128, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
	}

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool all_of(const simd_mask<T, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

#ifdef DPM_HAS_SSE4_1
		if constexpr (ext::native_data_size_v<mask_t> == 1)
		{
			const auto vm = detail::maskone<T>(mask_data[0], mask_t::size());
			return _mm_test_all_ones(std::bit_cast<__m128i>(vm));
		}
#endif
		auto result = detail::setones<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskone<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_and(result, vm);
		}
		return detail::movemask<T>(result) == sizeof(T) == 2 ? detail::fill_bits<4>() : detail::fill_bits<sizeof(T)>();
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool any_of(const simd_mask<T, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setzero<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_or(result, vm);
		}
#ifdef DPM_HAS_SSE4_1
		const auto vi = std::bit_cast<__m128i>(result);
		return !_mm_testz_si128(vi, vi);
#else
		return detail::movemask<T>(result);
#endif
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool none_of(const simd_mask<T, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setzero<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_or(result, vm);
		}
#ifdef DPM_HAS_SSE4_1
		const auto vi = std::bit_cast<__m128i>(result);
		return _mm_testz_si128(vi, vi);
#else
		return !detail::movemask<T>(result);
#endif
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool some_of(const simd_mask<T, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = detail::setzero<T>(), all_mask = detail::setones<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = mask_data[i];
			const auto vmz = detail::maskzero<T>(vm, mask_t::size() - i * sizeof(vm) / sizeof(T));
			const auto vmo = detail::maskone<T>(vm, mask_t::size() - i * sizeof(vm) / sizeof(T));

			all_mask = detail::bit_and(all_mask, vmo);
			any_mask = detail::bit_or(any_mask, vmz);
		}
#ifdef DPM_HAS_SSE4_1
		const auto any_vi = std::bit_cast<__m128i>(any_mask);
		const auto all_vi = std::bit_cast<__m128i>(all_mask);
		return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
		return detail::movemask<T>(any_mask) && detail::movemask<T>(all_mask) != sizeof(T) == 2 ? detail::fill_bits<4>() : detail::fill_bits<sizeof(T)>();
#endif
	}

	/** Returns the number of `true` elements of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t popcount(const simd_mask<T, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(mask_data[i]) / sizeof(T));
			result += std::popcount(detail::movemask<T>(vm));
		}

		if constexpr (sizeof(T) == 2) result /= 2;
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_first_set(const simd_mask<T, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<T, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = detail::movemask<T>(mask_data[i]);
			if (bits) return std::countr_zero(bits) + i * sizeof(T) <= 2 ? 1 : sizeof(T);
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_last_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::overload_128<T, N, A>
	{
		using mask_t = simd_mask<float, detail::avec<N, A>>;
		const auto mask_data = ext::to_native_data(mask);

		for (std::size_t i = ext::native_data_size_v<mask_t>, k; (k = i--) != 0;)
		{
			auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(mask_data[i]));
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

	template<typename T, std::size_t N, std::size_t Align> requires detail::overload_128<T, N, Align>
	class const_where_expression<simd_mask<T, detail::avec<N, Align>>, simd_mask<T, detail::avec<N, Align>>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

	protected:
		constexpr static auto vector_extent = 16 / sizeof(T);

		using mask_t = simd_mask<T, detail::avec<N, Align>>;
		using value_type = bool;

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(mask_t mask, mask_t &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(mask_t mask, const mask_t &data) noexcept : m_mask(mask), m_data(const_cast<mask_t &>(data)) {}

		/** Copies selected elements to \a mem. */
		template<typename Flags>
		DPM_FORCEINLINE void copy_to(value_type *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			const auto v_mask = ext::to_native_data(m_mask);
			const auto v_data = ext::to_native_data(m_data);
			for (std::size_t i = 0; i < mask_t::size(); i += vector_extent)
				mask_invoke(v_data[i / vector_extent], [&](std::size_t off, bool b)
				{
					if (v_mask[i / vector_extent][off]) mem[i + off] = b;
				}, mask_t::size() - i);
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<typename T, std::size_t N, std::size_t Align> requires detail::overload_128<T, N, Align>
	class where_expression<simd_mask<T, detail::avec<N, Align>>, simd_mask<T, detail::avec<N, Align>>>
			: public const_where_expression<simd_mask<T, detail::avec<N, Align>>, simd_mask<T, detail::avec<N, Align>>>
	{
		using base_expr = const_where_expression<simd_mask<T, detail::avec<N, Align>>, simd_mask<T, detail::avec<N, Align>>>;
		using value_type = typename base_expr::value_type;
		using mask_t = typename base_expr::mask_t;

		using base_expr::vector_extent;
		using base_expr::m_mask;
		using base_expr::m_data;

	public:
		using base_expr::const_where_expression;

		template<typename U>
		DPM_FORCEINLINE void operator=(U &&value) && noexcept requires std::is_convertible_v<U, value_type>
		{
			m_data = ext::blend(m_data, mask_t{std::forward<U>(value)}, m_mask);
		}

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
			for (std::size_t i = 0; i < mask_t::size(); i += vector_extent)
			{
				ext::native_data_type_t<mask_t> tmp = {};
				detail::mask_copy<T>(tmp, mem, mask_t::size() - i);
				v_data[i / vector_extent] = detail::bit_and(tmp, v_mask[i / vector_extent]);
			}
		}
	};

	DPM_DECLARE_EXT_NAMESPACE
	{
#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE simd_mask<T, detail::avec<N, A>> blend(
				const simd_mask<T, detail::avec<N, A>> &a,
				const simd_mask<T, detail::avec<N, A>> &b,
				const simd_mask<T, detail::avec<N, A>> &m)
		noexcept requires detail::overload_128<T, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd_mask<T, detail::avec<N, A>>>;
			simd_mask<T, detail::avec<N, A>> result = {};
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < data_size; ++i) result_data[i] = detail::blendv(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif
	}
}