/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "m128/type.hpp"
#include "m256/type.hpp"

namespace dpm
{
	namespace detail
	{
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		struct native_access<x86_mask<T, N, A>>
		{
			using mask_t = x86_mask<T, N, A>;
			static std::span<ext::native_data_type_t<mask_t>, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			static std::span<const ext::native_data_type_t<mask_t>, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
		template<typename T, std::size_t N, std::size_t A> requires detail::x86_overload_any<T, N, A>
		struct native_access<x86_simd<T, N, A>>
		{
			using simd_t = x86_simd<T, N, A>;
			static std::span<ext::native_data_type_t<simd_t>, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			static std::span<const ext::native_data_type_t<simd_t>, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};

		template<typename To, typename ToAbi, typename From, typename FromAbi>
		DPM_FORCEINLINE void cast_impl(simd<To, ToAbi> &to, const simd<From, FromAbi> &from) noexcept
		{
			const auto from_data = reinterpret_cast<const From *>(ext::to_native_data(from).data());
			constexpr auto from_align = alignof(decltype(from));
			constexpr auto to_align = alignof(decltype(to));

			if constexpr (to_align > from_align)
				to.copy_from(from_data, overaligned<to_align>);
			else if constexpr (to_align < from_align)
				to.copy_from(from_data, element_aligned);
			else
				to.copy_from(from_data, vector_aligned);
		}

		template<std::size_t I, typename T, typename OutAbi, typename XAbi, typename... Abis>
		DPM_FORCEINLINE void concat_impl(simd_mask<T, OutAbi> &out, const simd_mask<T, XAbi> &x, const simd_mask<T, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<T *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(ext::native_data_type_t<simd_mask<T, OutAbi>>) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd_mask<T, XAbi>::size()>(out, rest...);
		}
		template<std::size_t I, typename T, typename OutAbi, typename XAbi, typename... Abis>
		DPM_FORCEINLINE void concat_impl(simd<T, OutAbi> &out, const simd<T, XAbi> &x, const simd<T, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<T *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(ext::native_data_type_t<simd<T, OutAbi>>) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd<T, XAbi>::size()>(out, rest...);
		}

		template<typename T, std::size_t J, std::size_t... Is, typename V>
		DPM_FORCEINLINE void shuffle_unwrap(std::index_sequence<Is...> is, V *dst, const V *src) noexcept
		{
			constexpr auto vector_extent = sizeof(V) / sizeof(T);
			if constexpr (sizeof...(Is) == vector_extent)
			{
				if constexpr (!detail::is_sequential<0, Is...>::value)
					dst[J] = shuffle<T>(reverse_sequence_t<Is...>{}, src);
				else
					dst[J] = src[J];
			}
			else if constexpr (sizeof...(Is) > vector_extent)
			{
				shuffle_unwrap<T, J>(extract_sequence_t<J * vector_extent, vector_extent, Is...>{}, dst, src);
				shuffle_unwrap<T, J + 1>(is, dst, src);
			}
			else
				shuffle_unwrap<T, J>(pad_sequence_t<vector_extent, Is...>{}, dst, src);
		}
	}

	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>>
	{
		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

	protected:
		constexpr static auto vector_extent = 16 / sizeof(T);

		using mask_t = detail::x86_mask<T, N, Align>;
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
				detail::mask_invoke(v_data[i / vector_extent], [&](std::size_t off, bool b)
				{
					if (v_mask[i / vector_extent][off]) mem[i + off] = b;
				}, mask_t::size() - i);
		}

	protected:
		mask_t m_mask;
		mask_t &m_data;
	};
	template<typename T, std::size_t N, std::size_t Align> requires detail::x86_overload_any<T, N, Align>
	class where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>> : public const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>>
	{
		using base_expr = const_where_expression<detail::x86_mask<T, N, Align>, detail::x86_mask<T, N, Align>>;
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
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_mask<T, N, A>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(const detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_mask<T, N, A>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> blend(
				const detail::x86_mask<T, N, A> &a,
				const detail::x86_mask<T, N, A> &b,
				const detail::x86_mask<T, N, A> &m)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			detail::x86_mask<T, N, A> result = {};
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
				result_data[i] = detail::blendv(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif

		/** Shuffles elements of mask \a x into a new mask according to the specified indices. */
		template<std::size_t... Is, typename T, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> shuffle(const detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
		{
			detail::x86_mask<T, N, A> result = {};
			auto result_data = to_native_data(result).data();
			const auto x_data = to_native_data(x).data();

			detail::shuffle_unwrap<T, 0>(std::index_sequence<Is...>{}, result_data, x_data);
			return result;
		}
	}

#pragma region "simd_mask operators"
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator&(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			result_data[i] = detail::bit_and(a_data[i], b_data[i]);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator^(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			result_data[i] = detail::bit_xor(a_data[i], b_data[i]);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator|(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			result_data[i] = detail::bit_or(a_data[i], b_data[i]);
		return result;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> &operator&=(detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		const auto b_data = ext::to_native_data(b);
		auto a_data = ext::to_native_data(a);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			a_data[i] = detail::bit_and(a_data[i], b_data[i]);
		return a;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> &operator^=(detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		const auto b_data = ext::to_native_data(b);
		auto a_data = ext::to_native_data(a);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			a_data[i] = detail::bit_xor(a_data[i], b_data[i]);
		return a;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> &operator|=(detail::x86_mask<T, N, A> &a, const detail::x86_mask<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		const auto b_data = ext::to_native_data(b);
		auto a_data = ext::to_native_data(a);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			a_data[i] = detail::bit_or(a_data[i], b_data[i]);
		return a;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!(const detail::x86_mask<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto x_data = ext::to_native_data(x);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			result_data[i] = detail::bit_not(x_data[i]);
		return result;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator&&(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		return a & b;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator||(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		return a | b;
	}

	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator==(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_mask<T, N, A>>; ++i)
			result_data[i] = detail::mask_eq<T>(a_data[i], b_data[i]);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!=(
			const detail::x86_mask<T, N, A> &a,
			const detail::x86_mask<T, N, A> &b)
	noexcept requires detail::x86_overload_any<T, N, A>
	{
		return !(a == b);
	}
#pragma endregion

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool all_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setones<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskone<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_and(result, vm);
		}
		return detail::movemask<T>(result) == detail::fill_bits<sizeof(T) * detail::movemask_bits_v<T>>();
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool any_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setzero<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_or(result, vm);
		}
		return detail::movemask<T>(result);
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool none_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto result = detail::setzero<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(result) / sizeof(T));
			result = detail::bit_or(result, vm);
		}
		return !detail::movemask<T>(result);
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE bool some_of(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		auto any_mask = detail::setzero<T>(), all_mask = detail::setones<T>();
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = mask_data[i];
			const auto n = mask_t::size() - i * sizeof(vm) / sizeof(T);
			const auto vmz = detail::maskzero<T>(vm, n);
			const auto vmo = detail::maskone<T>(vm, n);

			all_mask = detail::bit_and(all_mask, vmo);
			any_mask = detail::bit_or(any_mask, vmz);
		}
		return detail::movemask<T>(any_mask) && detail::movemask<T>(all_mask) != detail::fill_bits<sizeof(T) * detail::movemask_bits_v<T>>();
	}

	/** Returns the number of `true` elements of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t popcount(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);

		std::size_t result = 0;
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto vm = detail::maskzero<T>(mask_data[i], mask_t::size() - i * sizeof(mask_data[i]) / sizeof(T));
			result += std::popcount(detail::movemask<T>(vm));
		}
		return result / detail::movemask_bits_v<T>;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_first_set(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);
		for (std::size_t i = 0; i < ext::native_data_size_v<mask_t>; ++i)
		{
			const auto bits = detail::movemask<T>(mask_data[i]);
			if (bits) return std::countr_zero(bits) / detail::movemask_bits_v<T> + i * sizeof(mask_data[i]) / sizeof(T);
		}
		DPM_UNREACHABLE();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE std::size_t find_last_set(const detail::x86_mask<T, N, A> &mask) noexcept requires detail::x86_overload_any<T, N, A>
	{
		using mask_t = detail::x86_mask<T, N, A>;
		const auto mask_data = ext::to_native_data(mask);
		for (std::size_t i = ext::native_data_size_v<mask_t>, j; (j = i--) != 0;)
		{
			constexpr auto vector_extent = sizeof(mask_data[i]) / sizeof(T);
			const auto bits = detail::movemask_l<T>(mask_data[i], mask_t::size() - i * vector_extent);
			if (bits) return (j * vector_extent - 1) - std::countl_zero(bits) / detail::movemask_bits_v<T>;
		}
		DPM_UNREACHABLE();
	}
#pragma endregion

#pragma region "simd_mask casts"
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, typename... Abis>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const simd_mask<T, Abis> &...values) noexcept requires ((detail::x86_simd_abi_any<Abis, T> && ...))
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>> result = {};
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const std::array<detail::x86_mask<T, N, A>, M> &values) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
	{
		using result_t = detail::x86_mask<T, N * M, A>;
		if constexpr (M == 1)
			return values[0];
		else
		{
			result_t result = {};
			for (std::size_t i = 0; i < M; ++i)
			{
				auto *data = reinterpret_cast<T *>(ext::to_native_data(result).data());
				if ((i * N) % sizeof(ext::native_data_type_t<result_t>) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}

	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask a copy of the `i + j * V::size()`th element from \a x.
	 * @note Size of \a x must be a multiple of `V::size()`. */
	template<typename V, std::size_t N, std::size_t A, typename U = typename V::simd_type::value_type>
	[[nodiscard]] DPM_FORCEINLINE auto split(const simd_mask<U, detail::avec<N, A>> &x) noexcept requires detail::can_split_mask<V, detail::avec<N, A>> && detail::x86_overload_any<U, N, A>
	{
		std::array<V, simd_size_v<U, detail::avec<N, A>> / V::size()> result = {};
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			const auto *data = reinterpret_cast<const U *>(ext::to_native_data(x).data());
			result[j].copy_from(data + j * V::size(), element_aligned);
		}
		return result;
	}
	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a x. */
	template<std::size_t N, typename T, std::size_t M, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE auto split_by(const detail::x86_mask<T, M, A> &x) noexcept requires (M % N == 0 && detail::x86_overload_any<T, M, A>)
	{
		return split<resize_simd_t<M / N, detail::x86_mask<T, M, A>>>(x);
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_simd<T, N, A>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE auto to_native_data(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
		{
			return detail::native_access<detail::x86_simd<T, N, A>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, std::size_t N, std::size_t A>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> blend(
				const detail::x86_simd<T, N, A> &a,
				const detail::x86_simd<T, N, A> &b,
				const detail::x86_mask<T, N, A> &m)
		noexcept requires detail::x86_overload_any<T, N, A>
		{
			constexpr auto data_size = native_data_size_v<detail::x86_simd<T, N, A>>;
			detail::x86_simd<T, N, A> result = {};
			auto result_data = to_native_data(result);
			const auto a_data = to_native_data(a);
			const auto b_data = to_native_data(b);
			const auto m_data = to_native_data(m);

			for (std::size_t i = 0; i < native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
				result_data[i] = detail::blendv(a_data[i], b_data[i], m_data[i]);
			return result;
		}
#endif

		/** Shuffles elements of vector \a x into a new vector according to the specified indices. */
		template<std::size_t... Is, typename T, std::size_t N, std::size_t A, std::size_t M = sizeof...(Is)>
		[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> shuffle(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
		{
			detail::x86_simd<T, N, A> result = {};
			auto result_data = to_native_data(result).data();
			const auto x_data = to_native_data(x).data();

			detail::shuffle_unwrap<T, 0>(std::index_sequence<Is...>{}, result_data, x_data);
			return result;
		}
	}

#pragma region "simd operators"
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator==(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_eq<T>(a_data[i], b_data[i]);
		return result;
	}
	template<typename T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator!=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_ne<T>(a_data[i], b_data[i]);
		return result;
	}

	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_gt<T>(a_data[i], b_data[i]);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_lt<T>(a_data[i], b_data[i]);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_ge<T>(a_data[i], b_data[i]);
		return result;
	}
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_le<T>(a_data[i], b_data[i]);
		return result;
	}

#ifdef DPM_HAS_SSE4_1
	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_mask<T, N, A> result = {};
		auto result_data = ext::to_native_data(result);
		const auto a_data = ext::to_native_data(a);
		const auto b_data = ext::to_native_data(b);

		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			result_data[i] = detail::cmp_gt<T>(a_data[i], b_data[i]);
		return result;
	}
	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return b > a;
	}
#endif

	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator>=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return (a > b) | (a == b);
	}
	template<std::signed_integral T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_mask<T, N, A> operator<=(const detail::x86_simd<T, N, A> &a, const detail::x86_simd<T, N, A> &b) noexcept requires detail::x86_overload_any<T, N, A>
	{
		return (a < b) | (a == b);
	}
#pragma endregion

#pragma region "simd casts"
	/** Implicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] DPM_FORCEINLINE auto simd_cast(const simd<U, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, U, detail::avec<N, A>> &&
	          detail::x86_overload_any<To, N, A> &&
	          detail::x86_overload_any<U, N, A>)
	{
		detail::cast_return_t<T, U, detail::avec<N, A>, simd<U, detail::avec<N, A>>::size()> result = {};
		detail::cast_impl(result, x);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] DPM_FORCEINLINE auto static_simd_cast(const simd<U, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, U, detail::avec<N, A>> &&
	          detail::x86_overload_any<To, N, A> &&
	          detail::x86_overload_any<U, N, A>)
	{
		detail::static_cast_return_t<T, U, detail::avec<N, A>, simd<U, detail::avec<N, A>>::size()> result = {};
		detail::cast_impl(result, x);
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD vector. */
	template<typename T, typename... Abis>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const simd<T, Abis> &...values) noexcept requires ((detail::x86_simd_abi_any<Abis, T> && ...))
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>> result = {};
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] DPM_FORCEINLINE auto concat(const std::array<detail::x86_simd<T, N, A>, M> &values) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
	{
		using result_t = detail::x86_simd<T, N * M, A>;
		if constexpr (M == 1)
			return values[0];
		else
		{
			result_t result = {};
			for (std::size_t i = 0; i < M; ++i)
			{
				auto *data = reinterpret_cast<T *>(ext::to_native_data(result).data());
				if ((i * N) % sizeof(ext::native_data_type_t<result_t>) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}

	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * V::size()`th element from \a x.
	 * @note Size of \a x must be a multiple of `V::size()`. */
	template<typename V, std::size_t N, std::size_t A, typename U = typename V::simd_type::value_type>
	[[nodiscard]] DPM_FORCEINLINE auto split(const simd<U, detail::avec<N, A>> &x) noexcept requires detail::can_split_mask<V, detail::avec<N, A>> && detail::x86_overload_any<U, N, A>
	{
		std::array<V, simd_size_v<U, detail::avec<N, A>> / V::size()> result = {};
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			const auto *data = reinterpret_cast<const U *>(ext::to_native_data(x).data());
			result[j].copy_from(data + j * V::size(), element_aligned);
		}
		return result;
	}
	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a x. */
	template<std::size_t N, typename T, std::size_t M, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE auto split_by(const detail::x86_simd<T, M, A> &x) noexcept requires (M % N == 0 && detail::x86_overload_any<T, M, A>)
	{
		return split<resize_simd_t<M / N, detail::x86_simd<T, M, A>>>(x);
	}
#pragma endregion
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif