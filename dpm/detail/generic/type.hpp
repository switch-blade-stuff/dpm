/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../../flags.hpp"
#include "../../utility.hpp"
#include "../alias.hpp"

#include "abi.hpp"

#include <algorithm>
#include <array>
#include <span>

namespace dpm
{
	template<typename T, typename Abi = simd_abi::native<T>>
	class simd_mask;

	template<typename T>
	using native_simd_mask = simd_mask<T, simd_abi::native<T>>;
	template<typename T, std::size_t N>
	using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

	template<typename T, typename U, typename Abi>
	struct rebind_simd<T, simd_mask<U, Abi>> { using type = simd_mask<T, simd_abi::deduce_t<T, simd_size_v<U, Abi>, Abi>>; };
	template<std::size_t N, typename T, typename Abi>
	struct resize_simd<N, simd_mask<T, Abi>> { using type = simd_mask<T, simd_abi::deduce_t<T, N, Abi>>; };

	template<typename T, typename Abi = simd_abi::native<T>>
	class simd;

	template<typename T>
	using native_simd = simd<T, simd_abi::native<T>>;
	template<typename T, std::size_t N>
	using fixed_size_simd = simd<T, simd_abi::fixed_size<N>>;

	template<typename T, typename U, typename Abi>
	struct rebind_simd<T, simd<U, Abi>> { using type = simd<T, simd_abi::deduce_t<T, simd_size_v<U, Abi>, Abi>>; };
	template<std::size_t N, typename T, typename Abi>
	struct resize_simd<N, simd<T, Abi>> { using type = simd<T, simd_abi::deduce_t<T, N, Abi>>; };

	namespace detail
	{
		/* Clang-tidy was having issues with friended concept-constrained functions and types, so this is a workaround. */
		template<typename>
		struct native_access;

		template<typename T>
		struct get_simd_value;
		template<typename T, typename Abi>
		struct get_simd_value<simd<T, Abi>> { using type = T; };
		template<typename T, typename Abi>
		struct get_simd_value<simd_mask<T, Abi>> { using type = bool; };

		template<typename T>
		struct get_simd_abi;
		template<typename T, typename Abi>
		struct get_simd_abi<simd<T, Abi>> { using type = Abi; };
		template<typename T, typename Abi>
		struct get_simd_abi<simd_mask<T, Abi>> { using type = Abi; };

		template<int, std::size_t, std::size_t...>
		struct is_sequential : std::false_type {};
		template<int Step, std::size_t I>
		struct is_sequential<Step, I, I> : std::true_type {};
		template<int Step, std::size_t I, std::size_t... Is>
		struct is_sequential<Step, I, I, Is...> : is_sequential<Step, I + static_cast<std::size_t>(Step), Is...> {};

		struct bool_wrapper
		{
			bool_wrapper() = delete;

			constexpr bool_wrapper(bool value) noexcept : value(value) {}
			constexpr operator bool() const noexcept { return value; }

			bool value;
		};

		template<std::size_t N, std::size_t I = 0, typename G>
		constexpr DPM_FORCEINLINE void generate_n(auto &data, G &&gen) noexcept
		{
			if constexpr (I != N)
			{
				data[I] = std::invoke(gen, std::integral_constant<std::size_t, I>());
				generate_n<N, I + 1>(data, std::forward<G>(gen));
			}
		}

		template<typename From, typename To, typename FromAbi, typename ToAbi>
		constexpr DPM_FORCEINLINE void copy_cast(const simd_mask<From, FromAbi> &from, simd_mask<To, ToAbi> &to) noexcept
		{
			if constexpr (std::same_as<simd_mask<From, FromAbi>, simd_mask<To, ToAbi>>)
				to = from;
			else
			{
				constexpr auto result_align = std::max(alignof(simd_mask<From, FromAbi>), alignof(simd_mask<To, ToAbi>));
				constexpr auto result_size = std::max(simd_mask<From, FromAbi>::size(), simd_mask<To, ToAbi>::size());

				alignas(result_align) std::array<bool, result_size> result_buff;
				from.copy_to(result_buff.data(), vector_aligned);
				to.copy_from(result_buff.data(), vector_aligned);
			}
		}
		template<typename From, typename To, typename FromAbi, typename ToAbi>
		constexpr DPM_FORCEINLINE void copy_cast(const simd<From, FromAbi> &from, simd<To, ToAbi> &to) noexcept
		{
			if constexpr (std::same_as<simd<From, FromAbi>, simd<To, ToAbi>>)
				to = from;
			else
			{
				constexpr auto result_align = std::max(alignof(simd<From, FromAbi>), alignof(simd<To, ToAbi>));
				constexpr auto result_size = std::max(simd<From, FromAbi>::size(), simd<To, ToAbi>::size());

				alignas(result_align) std::array<To, result_size> result_buff;
				from.copy_to(result_buff.data(), vector_aligned);
				to.copy_from(result_buff.data(), vector_aligned);
			}
		}

		template<std::size_t J, std::size_t I, std::size_t... Is, typename T, typename FromAbi, typename ToAbi>
		constexpr DPM_FORCEINLINE void shuffle_impl(const simd_mask<T, FromAbi> &from, simd_mask<T, ToAbi> &to) noexcept
		{
			static_assert(J < simd_mask<T, ToAbi>::size() && I < simd_mask<T, FromAbi>::size());
			if constexpr (sizeof...(Is) != 0) shuffle_impl<J + 1, Is...>(from, to);
			to[J] = from[I];
		}
		template<std::size_t J, std::size_t I, std::size_t... Is, typename T, typename FromAbi, typename ToAbi>
		constexpr DPM_FORCEINLINE void shuffle_impl(const simd<T, FromAbi> &from, simd<T, ToAbi> &to) noexcept
		{
			static_assert(J < simd<T, ToAbi>::size() && I < simd<T, FromAbi>::size());
			if constexpr (sizeof...(Is) != 0) shuffle_impl<J + 1, Is...>(from, to);
			to[J] = from[I];
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T>
		struct native_data_type : detail::get_simd_value<T> {};
		template<typename T>
		struct native_data_size : simd_size<typename detail::get_simd_value<T>::type, typename detail::get_simd_abi<T>::type> {};
	}

	/** Returns \a x. */
	[[nodiscard]] constexpr bool all_of(detail::bool_wrapper x) noexcept { return x; }
	/** @copydoc any_of */
	[[nodiscard]] constexpr bool any_of(detail::bool_wrapper x) noexcept { return x; }
	/** Returns the negation of \a x. */
	[[nodiscard]] constexpr bool none_of(detail::bool_wrapper x) noexcept { return !x; }
	/** Returns `false`. */
	[[nodiscard]] constexpr bool some_of([[maybe_unused]] detail::bool_wrapper value) noexcept { return false; }
	/** Returns the integral representation of \a x. */
	[[nodiscard]] constexpr std::size_t reduce_count(detail::bool_wrapper x) noexcept { return static_cast<std::size_t>(x); }

	/** Returns `0`. */
	[[nodiscard]] constexpr std::size_t reduce_min_index([[maybe_unused]] detail::bool_wrapper value) noexcept { return 0; }
	/** @copydoc reduce_max_index */
	[[nodiscard]] constexpr std::size_t reduce_max_index([[maybe_unused]] detail::bool_wrapper value) noexcept { return 0; }

	/** Equivalent to `m ? a : b`. */
	template<typename T>
	[[nodiscard]] constexpr T blend(detail::bool_wrapper m, const T &a, const T &b) noexcept { return m ? a : b; }

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Equivalent to `shuffle<Is...>(fixed_size_simd<T, 1>(x))`. */
		template<std::size_t... Is, typename T>
		[[nodiscard]] constexpr fixed_size_simd<T, sizeof...(Is)> shuffle(const T &x) noexcept { return shuffle<Is...>(fixed_size_simd<T, 1>{x}); }
	}

	/** @brief Type representing a data-parallel mask vector type.
	 * @tparam T Value type stored by the SIMD mask.
	 * @tparam Abi ABI used to select implementation of the SIMD mask. */
	template<typename T, typename Abi>
	class simd_mask
	{
		friend detail::native_access<simd_mask>;

	public:
		using abi_type = Abi;
		using value_type = bool;
		using reference = value_type &;
		using simd_type = simd<T, abi_type>;
		using mask_type = simd_mask<T, abi_type>;

		static constexpr auto size = std::integral_constant<std::size_t, abi_type::size>{};

	private:
		constexpr static auto alignment = std::max(abi_type::alignment == SIZE_MAX ? 0 : abi_type::alignment, alignof(bool[size()]));

	public:
		constexpr simd_mask() noexcept = default;
		/** Initializes the underlying elements with \a value. */
		constexpr simd_mask(value_type value) noexcept { std::fill_n(_data, size(), value); }
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		constexpr simd_mask(G &&gen) noexcept { detail::generate_n<size()>(_data, std::forward<G>(gen)); }

		/** Initializes the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd_mask(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::convertible_to<std::iter_value_t<I>, value_type> { copy_from(mem, Flags{}); }
		/** Initializes the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd_mask(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::convertible_to<std::iter_value_t<I>, value_type> { copy_from(mem, m, Flags{}); }

		template<typename U, typename OtherAbi> requires(simd_size_v<T, OtherAbi> == size())
		constexpr operator simd_mask<U, OtherAbi>() const noexcept { return simd_mask<U, OtherAbi>{_data, element_aligned}; }

		/** Copies the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				_data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		/** Copies the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) _data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, const mask_type &m, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		[[nodiscard]] constexpr reference operator[](std::size_t i) & noexcept { return _data[i]; }
		[[nodiscard]] constexpr value_type operator[](std::size_t i) const & noexcept { return _data[i]; }

	private:
		alignas(alignment) value_type _data[size()];
	};

	namespace detail
	{
		template<typename T, typename Abi>
		struct native_access<simd_mask<T, Abi>>
		{
			using mask_t = simd_mask<T, Abi>;

			[[nodiscard]] static std::span<bool, simd_size_v<T, Abi>> to_native_data(mask_t &x) noexcept { return {x._data}; }
			[[nodiscard]] static std::span<const bool, simd_size_v<T, Abi>> to_native_data(const mask_t &x) noexcept { return {x._data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying values of \a x. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr auto to_native_data(simd_mask<T, Abi> &x) noexcept { return detail::native_access<simd_mask<T, Abi>>::to_native_data(x); }
		/** Returns a constant span of the underlying values of \a x. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr auto to_native_data(const simd_mask<T, Abi> &x) noexcept { return detail::native_access<simd_mask<T, Abi>>::to_native_data(x); }

		/** Shuffles elements of mask \a x into a new mask according to the specified indices. ABI of the resulting mask is deduced via `simd_abi::deduce_value<T, sizeof...(Is), Abi>`. */
		template<std::size_t I, std::size_t... Is, typename T, typename Abi>
		[[nodiscard]] constexpr simd_mask<T, simd_abi::deduce_t<T, sizeof...(Is) + 1, Abi>> shuffle(const simd_mask<T, Abi> &x)
		{
			using result_t = simd_mask<T, simd_abi::deduce_t<T, sizeof...(Is) + 1, Abi>>;
			if constexpr (detail::is_sequential<1, 0, I, Is...>::value && result_t::size() == simd_mask<T, Abi>::size())
				return result_t{x};
			else if constexpr (sizeof...(Is) == 0 || ((Is == I) && ...))
				return result_t{x[I]};
			else
			{
				result_t result = {};
				detail::shuffle_impl<0, I, Is...>(x, result);
				return result;
			}
		}
	}

	/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a a are selected if the corresponding element of \a m evaluates to `true`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> blend(const simd_mask<T, Abi> &m, const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < m.size; ++i)
			result[i] = m[i] ? a[i] : b[i];
		return {result.data(), element_aligned};
	}

#pragma region "simd_mask operators"
	/** Preforms a bitwise NOT on the elements of the mask \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator~(const simd_mask<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = ~x[i];
		return {result.data(), element_aligned};
	}
	/** Preforms a bitwise AND on the elements of the masks \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator&(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] && b[i];
		return {result.data(), element_aligned};
	}
	/** Preforms a bitwise OR on the elements of the masks \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator|(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] || b[i];
		return {result.data(), element_aligned};
	}
	/** Preforms a bitwise XOR on the elements of the masks \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator^(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b[i];
		return {result.data(), element_aligned};
	}

	/** Returns a copy of the mask \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator+(const simd_mask<T, Abi> &x) noexcept { return x; }
	/** Returns an inverted copy of the mask \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator-(const simd_mask<T, Abi> &x) noexcept { return ~x; }

	/** Preforms a bitwise AND on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	constexpr simd_mask<T, Abi> &operator&=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a & b; }
	/** Preforms a bitwise OR on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	constexpr simd_mask<T, Abi> &operator|=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a | b; }
	/** Preforms a bitwise XOR on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	constexpr simd_mask<T, Abi> &operator^=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a ^ b; }

	/** Preforms a logical AND on the elements of the masks \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator&&(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] && b[i];
		return {result.data(), element_aligned};
	}
	/** Preforms a logical OR on the elements of the masks \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator||(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] || b[i];
		return {result.data(), element_aligned};
	}

	/** Inverts elements of mask \a x . */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator!(const simd_mask<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = !x[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of masks \a a and \a b for equality. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator==(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] == b[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of masks \a a and \a b for inequality. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator!=(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] != b[i];
		return {result.data(), element_aligned};
	}
#pragma endregion

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr bool all_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool result = true;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result &= mask[i];
		return result;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr bool any_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool result = false;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result |= mask[i];
		return result;
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr bool none_of(const simd_mask<T, Abi> &mask) noexcept { return !any_of(mask); }
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr bool some_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool any_true = false, all_true = true;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
		{
			const auto b = mask[i];
			all_true &= b;
			any_true |= b;
		}
		return any_true && !all_true;
	}

	/** Returns the number of `true` elements of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr std::size_t reduce_count(const simd_mask<T, Abi> &mask) noexcept
	{
		std::size_t result = 0;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result += mask[i];
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr std::size_t reduce_min_index(const simd_mask<T, Abi> &mask) noexcept
	{
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			if (mask[i]) return i;
		return simd_mask<T, Abi>::size();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr std::size_t reduce_max_index(const simd_mask<T, Abi> &mask) noexcept
	{
		for (std::size_t i = simd_mask<T, Abi>::size(); i-- > 0;)
			if (mask[i]) return i;
		return simd_mask<T, Abi>::size();
	}
#pragma endregion

#pragma region "simd_mask casts"
	/** Converts SIMD mask \a x to it's fixed-size equivalent for value type \a T. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr fixed_size_simd_mask<T, simd_size_v<T, Abi>> to_fixed_size(const simd_mask<T, Abi> &x) noexcept { return {x}; }
	/** Converts SIMD mask \a x to it's native ABI equivalent for value type \a T. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr native_simd_mask<T> to_native(const simd_mask<T, Abi> &x) noexcept { return {x}; }
	/** Converts SIMD mask \a x to it's compatible ABI equivalent for value type \a T. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T> to_compatible(const simd_mask<T, Abi> &x) noexcept { return {x}; }
#pragma endregion

	/** @brief Type representing a data-parallel arithmetic vector.
	 * @tparam T Value type stored by the SIMD vector.
	 * @tparam Abi ABI used to select implementation of the SIMD vector. */
	template<typename T, typename Abi>
	class simd
	{
		friend struct detail::native_access<simd>;

	public:
		using abi_type = Abi;
		using value_type = T;
		using reference = value_type &;
		using mask_type = simd_mask<T, abi_type>;

		static constexpr auto size = std::integral_constant<std::size_t, abi_type::size>{};

	private:
		constexpr static auto alignment = std::max(abi_type::alignment == SIZE_MAX ? 0 : abi_type::alignment, alignof(value_type[size()]));

	public:
		constexpr simd() noexcept = default;
		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		constexpr simd(U &&value) noexcept { std::fill_n(_data, size(), static_cast<value_type>(std::forward<U>(value))); }
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		constexpr simd(G &&gen) noexcept { detail::generate_n<size()>(_data, std::forward<G>(gen)); }

		/** Initializes the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::convertible_to<std::iter_value_t<I>, value_type> { copy_from(mem, Flags{}); }
		/** Initializes the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr simd(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::convertible_to<std::iter_value_t<I>, value_type> { copy_from(mem, m, Flags{}); }

		template<typename U, typename OtherAbi> requires(simd_size_v<T, OtherAbi> == size() && std::is_convertible_v<T, U>)
		constexpr explicit(!std::convertible_to<T, U>) operator simd<U, OtherAbi>() const noexcept { return simd<U, OtherAbi>{_data, element_aligned}; }

		/** Copies the underlying elements from \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				_data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		/** Copies the underlying elements from \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_from(I mem, const mask_type &m, Flags) noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<std::iter_value_t<I>, value_type>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) _data[i] = static_cast<value_type>(mem[i]);
		}
		/** Copies the underlying elements to \a mem if the corresponding element of mask \a m evaluates to `true`. */
		template<std::contiguous_iterator I, typename Flags>
		constexpr void copy_to(I mem, const mask_type &m, Flags) const noexcept requires is_simd_flag_type_v<Flags> && std::is_convertible_v<value_type, std::iter_value_t<I>>
		{
			for (std::size_t i = 0; i < size(); ++i)
				if (m[i]) mem[i] = static_cast<std::iter_value_t<I>>(_data[i]);
		}

		[[nodiscard]] constexpr reference operator[](std::size_t i) & noexcept { return _data[i]; }
		[[nodiscard]] constexpr value_type operator[](std::size_t i) const & noexcept { return _data[i]; }

	private:
		alignas(alignment) value_type _data[size()];
	};

	namespace detail
	{
		template<typename T, typename Abi>
		struct native_access<simd<T, Abi>>
		{
			using simd_t = simd<T, Abi>;

			[[nodiscard]] static std::span<T, simd_size_v<T, Abi>> to_native_data(simd_t &x) noexcept { return {x._data}; }
			[[nodiscard]] static std::span<const T, simd_size_v<T, Abi>> to_native_data(const simd_t &x) noexcept { return {x._data}; }
		};

		template<typename T>
		struct deduce_cast { using type = T; };
		template<typename T> requires is_simd_v<T>
		struct deduce_cast<T> { using type = typename T::value_type; };

		template<typename T, typename U, typename Abi, std::size_t N>
		struct cast_return { using type = std::conditional_t<std::same_as<T, U>, simd<T, Abi>, simd<T, simd_abi::deduce_t<T, N, Abi, simd_abi::fixed_size<N>>>>; };
		template<typename T, typename U, typename Abi, std::size_t N> requires is_simd_v<T>
		struct cast_return<T, U, Abi, N> { using type = T; };

		template<typename T, typename U, typename Abi, std::size_t N>
		using cast_return_t = typename cast_return<T, U, Abi, N>::type;

		template<typename T, typename U, typename Abi, std::size_t N>
		struct static_cast_return { using type = std::conditional_t<std::same_as<T, U>, simd<T, Abi>, simd<T, simd_abi::deduce_t<T, N, Abi, simd_abi::fixed_size<N>>>>; };
		template<std::integral T, std::integral U, typename Abi, std::size_t N>
		struct static_cast_return<T, U, Abi, N> { using type = std::conditional_t<std::same_as<T, U> || std::same_as<std::make_signed_t<T>, std::make_signed_t<U>>, simd<T, Abi>, simd<T, simd_abi::deduce_t<T, N, Abi, simd_abi::fixed_size<N>>>>; };
		template<typename T, typename U, typename Abi, std::size_t N> requires is_simd_v<T>
		struct static_cast_return<T, U, Abi, N> { using type = T; };

		template<typename T, typename U, typename Abi, std::size_t N>
		using static_cast_return_t = typename static_cast_return<T, U, Abi, N>::type;

		template<typename T, typename U, typename Abi>
		struct equal_cast_size : std::bool_constant<simd<U, Abi>::size() == T::size()> {};
		template<typename T, typename U, typename Abi> requires(!is_simd_v<T>)
		struct equal_cast_size<T, U, Abi> : std::true_type {};

		template<typename T, typename U, typename Abi>
		concept valid_simd_cast = std::is_convertible_v<U, typename detail::deduce_cast<T>::type> && detail::equal_cast_size<T, U, Abi>::value;
		template<typename T, typename U, typename Abi>
		concept valid_simd_static_cast = std::convertible_to<U, typename detail::deduce_cast<T>::type> && detail::equal_cast_size<T, U, Abi>::value;

		template<typename T, std::size_t N, typename Op>
		[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_impl(const std::array<T, N> &data, Op binary_op)
		{
			if constexpr (N == 1)
				return data[0];
			else if constexpr (N != 2)
			{
				std::array<T, N - N / 2> b;
				std::array<T, N / 2> a;

				/* Separate `value` into halves and reduce them separately. */
				for (std::size_t i = 0; i < a.size(); ++i)
					a[i] = data[0];
				for (std::size_t i = 0, j = a.size(); i < b.size(); ++i, ++j)
					b[i] = data[j];

				return reduce_impl(std::array{reduce_impl(a, binary_op), reduce_impl(b, binary_op)}, binary_op);
			}
			else if constexpr (!std::is_invocable_r_v<T, Op, T, T>)
				return std::invoke(binary_op, simd<T, simd_abi::scalar>{data[0]}, simd<T, simd_abi::scalar>{data[1]})[0];
			else
				return std::invoke(binary_op, data[0], data[1]);
		}
		template<std::size_t N, typename T, typename Abi, typename Op>
		[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_impl(const simd<T, Abi> &x, Op binary_op)
		{
			alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> buff;
			x.copy_to(buff.data(), vector_aligned);
			return reduce_impl(buff, binary_op);
		}

		template<std::size_t N, std::unsigned_integral T>
		constexpr DPM_FORCEINLINE T lsr_impl(T x) noexcept
		{
			if constexpr (T{-1} >> N == T{-1})
				return (x >> N) & (T{1} << (std::numeric_limits<T>::digits - 1));
			else
				return x >> N;
		}
		template<std::size_t N, std::signed_integral T>
		constexpr DPM_FORCEINLINE T lsr_impl(T x) noexcept
		{
			if constexpr (T{-1} >> N == T{-1})
				return lsr_impl<N>(static_cast<std::make_unsigned_t<T>>(x));
			else
				return x >> N;
		}
		template<std::size_t N, typename T>
		constexpr DPM_FORCEINLINE T asr_impl(T x) noexcept
		{
			if constexpr (T{-1} >> N != T{-1})
				return x < 0 ? ~(~x >> N) : x >> N;
			else
				return x >> N;
		}
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying elements of \a x. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr auto to_native_data(simd<T, Abi> &x) noexcept
		{
			return detail::native_access<simd<T, Abi>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying elements of \a x. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr auto to_native_data(const simd<T, Abi> &x) noexcept
		{
			return detail::native_access<simd<T, Abi>>::to_native_data(x);
		}

		/** Shuffles elements of vector \a x into a new vector according to the specified indices. ABI of the resulting vector is deduced via `simd_abi::deduce_value<T, sizeof...(Is), Abi>`. */
		template<std::size_t I, std::size_t... Is, typename T, typename Abi>
		[[nodiscard]] constexpr simd<T, simd_abi::deduce_t<T, sizeof...(Is) + 1, Abi>> shuffle(const simd<T, Abi> &x) noexcept
		{
			using result_t = simd<T, simd_abi::deduce_t<T, sizeof...(Is) + 1, Abi>>;
			if constexpr (detail::is_sequential<1, 0, I, Is...>::value && result_t::size() == simd<T, Abi>::size())
				return result_t{x};
			else if constexpr (sizeof...(Is) == 0 || ((Is == I) && ...))
				return result_t{x[I]};
			else
			{
				result_t result = {};
				detail::shuffle_impl<0, I, Is...>(x, result);
				return result;
			}
		}
	}

	/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a a are selected if the corresponding element of \a m evaluates to `true`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> blend(const simd_mask<T, Abi> &m, const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < m.size; ++i)
			result[i] = m[i] ? a[i] : b[i];
		return {result.data(), vector_aligned};
	}

#pragma region "simd operators"
	/** Returns a copy of vector \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator+(const simd<T, Abi> &x) noexcept { return x; }
	/** Negates elements of vector \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator-(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = -x[i];
		return {result.data(), vector_aligned};
	}

	/** Increments elements of vector \a x, and returns the copy of the original vector. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator++(simd<T, Abi> &x, int) noexcept requires(requires(T v){ v++; })
	{
		auto tmp = x;
		operator++(x);
		return tmp;
	}
	/** Decrements elements of vector \a x, and returns the copy of the original vector. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator--(simd<T, Abi> &x, int) noexcept requires(requires(T v){ v--; })
	{
		auto tmp = x;
		operator--(x);
		return tmp;
	}
	/** Increments elements of vector \a x, and returns reference to it. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> &operator++(simd<T, Abi> &x) noexcept requires(requires(T v){ ++v; }) { return x = x + 1; }
	/** Decrements elements of vector \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> &operator--(simd<T, Abi> &x) noexcept requires(requires(T v){ --v; }) { return x = x - 1; }

	/** Adds elements of vector \a b to elements of vector \a a. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator+(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l + r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] + b[i];
		return {result.data(), vector_aligned};
	}
	/** Subtracts elements of vector \a b from elements of vector \a a. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator-(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l - r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] - b[i];
		return {result.data(), vector_aligned};
	}
	/** Adds elements of vector \a b to elements of vector \a a, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator+=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l += r; }) { return a = a + b; }
	/** Subtracts elements of vector \a b from elements of vector \a a, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator-=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l -= r; }) { return a = a - b; }

	/** Adds scalar \a b to elements of vector \a a. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator+(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l + r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] + b;
		return {result.data(), vector_aligned};
	}
	/** Subtracts scalar \a b from elements of vector \a a. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator-(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l - r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] - b;
		return {result.data(), vector_aligned};
	}
	/** Adds scalar \a b to elements of vector \a a, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator+=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l += r; }) { return a = a + b; }
	/** Subtracts scalar \a b from elements of vector \a a, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator-=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l -= r; }) { return a = a - b; }

	/** Multiplies elements of vector \a a by elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator*(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l * r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] * b[i];
		return {result.data(), vector_aligned};
	}
	/** Divides elements of vector \a a by elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator/(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l / r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] / b[i];
		return {result.data(), vector_aligned};
	}
	/** Preforms a modulo operation of elements of vector \a a by elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator%(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l % r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] % b[i];
		return {result.data(), vector_aligned};
	}
	/** Multiplies elements of vector \a a by elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator*=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l *= r; }) { return a = a * b; }
	/** Divides elements of vector \a a by elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator/=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l /= r; }) { return a = a / b; }
	/** Preforms a modulo operation of elements of vector \a a by elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator%=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l %= r; }) { return a = a % b; }

	/** Multiplies elements of vector \a a by scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator*(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l * r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] * b;
		return {result.data(), vector_aligned};
	}
	/** Divides elements of vector \a a by scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator/(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l / r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] / b;
		return {result.data(), vector_aligned};
	}
	/** Preforms a modulo operation of elements of vector \a a by scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator%(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l % r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] % b;
		return {result.data(), vector_aligned};
	}
	/** Multiplies elements of vector \a a by scalar \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator*=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l *= r; }) { return a = a * b; }
	/** Divides elements of vector \a a by scalar \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator/=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l /= r; }) { return a = a / b; }
	/** Preforms a modulo operation of elements of vector \a a by scalar \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator%=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l %= r; }) { return a = a % b; }

	/** Preforms a bitwise NOT on elements of vector \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator~(const simd<T, Abi> &x) noexcept requires(requires(T v){ ~v; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = ~x[i];
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise AND between elements of vector \a a and elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator&(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l & r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] & b[i];
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise OR between elements of vector \a a and elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator|(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l | r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] | b[i];
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise XOR between elements of vector \a a and elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator^(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l ^ r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b[i];
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise AND between elements of vector \a a and elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator&=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l &= r; }) { return a = a & b; }
	/** Preforms a bitwise OR between elements of vector \a a and elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator|=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l |= r; }) { return a = a | b; }
	/** Preforms a bitwise XOR between elements of vector \a a and elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator^=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l ^= r; }) { return a = a ^ b; }

	/** Preforms a bitwise AND between elements of vector \a a and scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator&(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l & r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] & b;
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise OR between elements of vector \a a and scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator|(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l | r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] | b;
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise XOR between elements of vector \a a and scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator^(const simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l ^ r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b;
		return {result.data(), vector_aligned};
	}
	/** Preforms a bitwise AND between elements of vector \a a and scalar \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator&=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l &= r; }) { return a = a & b; }
	/** Preforms a bitwise OR between elements of vector \a a and scalar \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator|=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l |= r; }) { return a = a | b; }
	/** Preforms a bitwise XOR between elements of vector \a a and scalar \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator^=(simd<T, Abi> &a, T b) noexcept requires(requires(T l, T r){ l ^= r; }) { return a = a ^ b; }

	/** Shifts elements of vector \a a left by the amount specified by elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator<<(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l << r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] << b[i];
		return {result.data(), vector_aligned};
	}
	/** Shifts elements of vector \a a right by the amount specified by elements of vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> operator>>(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l >> r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >> b[i];
		return {result.data(), vector_aligned};
	}
	/** Shifts elements of vector \a a left by the amount specified by elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator<<=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l <<= r; }) { return a = a >> b; }
	/** Shifts elements of vector \a a right by the amount specified by elements of vector \a b, and returns reference to \a a. */
	template<typename T, typename Abi>
	constexpr simd<T, Abi> &operator>>=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires(requires(T l, T r){ l >>= r; }) { return a = a << b; }

	/** Shifts elements of vector \a a left by \a n. */
	template<typename T, typename Abi, std::integral I>
	[[nodiscard]] constexpr simd<T, Abi> operator<<(const simd<T, Abi> &a, I n) noexcept requires(requires(T l, I r){ l << r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] << n;
		return {result.data(), vector_aligned};
	}
	/** Shifts elements of vector \a a right by \a n. */
	template<typename T, typename Abi, std::integral I>
	[[nodiscard]] constexpr simd<T, Abi> operator>>(const simd<T, Abi> &a, I n) noexcept requires(requires(T l, I r){ l >> r; })
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >> n;
		return {result.data(), vector_aligned};
	}
	/** Shifts elements of vector \a a right by \a n, and returns reference to \a a. */
	template<typename T, typename Abi, std::integral I>
	constexpr simd<T, Abi> &operator>>=(simd<T, Abi> &a, I n) noexcept requires(requires(T l, I r){ l >>= r; }) { return a = a >> n; }
	/** Shifts elements of vector \a a left by \a n, and returns reference to \a a. */
	template<typename T, typename Abi, std::integral I>
	constexpr simd<T, Abi> &operator<<=(simd<T, Abi> &a, I n) noexcept requires(requires(T l, I r){ l <<= r; }) { return a = a << n; }

	/** Converts elements of vector \a x to booleans. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator!(const simd<T, Abi> &x) noexcept requires(requires(T v) { !v; })
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = !x[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of vectors \a a and \a b for equality. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator==(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] == b[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of vectors \a a and \a b for less-than or equal. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator<=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] <= b[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of vectors \a a and \a b for greater-than or equal. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator>=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >= b[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of vectors \a a and \a b for less-than. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator<(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] < b[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of vectors \a a and \a b for greater-than. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator>(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] > b[i];
		return {result.data(), element_aligned};
	}
	/** Compares elements of vectors \a a and \a b for equality. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd_mask<T, Abi> operator!=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] != b[i];
		return {result.data(), element_aligned};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Logically shifts elements of vector \a x left by a constant number of bits \a N. */
		template<std::size_t N, std::integral T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> lsl(const simd<T, Abi> &x) noexcept requires(N < std::numeric_limits<T>::digits)
		{
			alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
				result[i] = x[i] << N;
			return {result.data(), vector_aligned};
		}
		/** Logically shifts elements of vector \a x right by a constant number of bits \a N. */
		template<std::size_t N, std::integral T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> lsr(const simd<T, Abi> &x) noexcept requires(N < std::numeric_limits<T>::digits)
		{
			alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
				result[i] = detail::lsr_impl<N>(x[i]);
			return {result.data(), vector_aligned};
		}

		/** Arithmetically shifts elements of vector \a x left by a constant number of bits \a N. */
		template<std::size_t N, std::signed_integral T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> asl(const simd<T, Abi> &x) noexcept requires(N < std::numeric_limits<T>::digits)
		{
			alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
				result[i] = x[i] << N;
			return {result.data(), vector_aligned};
		}
		/** Arithmetically shifts elements of vector \a x right by a constant number of bits \a N. */
		template<std::size_t N, std::signed_integral T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> asr(const simd<T, Abi> &x) noexcept requires(N < std::numeric_limits<T>::digits)
		{
			alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
				result[i] = detail::asr_impl<N>(x[i]);
			return {result.data(), vector_aligned};
		}
	}
#pragma endregion

#pragma region "simd reductions"
	/** Calculates a reduction of all elements from \a x using \a binary_op. */
	template<typename T, typename Abi, typename Op = std::plus<>>
	[[nodiscard]] constexpr T reduce(const simd<T, Abi> &x, Op binary_op = {}) noexcept(std::is_nothrow_invocable_v<Op, T, T>) { return dpm::detail::reduce_impl<simd_size_v<T, Abi>>(x, binary_op); }

	/** Finds the minimum of all elements (horizontal minimum) in \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_min(const simd<T, Abi> &x) noexcept { return reduce(x, [](T a, T b) { return std::min(a, b); }); }
	/** Finds the maximum of all elements (horizontal maximum) in \a x. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE T reduce_max(const simd<T, Abi> &x) noexcept { return reduce(x, [](T a, T b) { return std::max(a, b); }); }

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Finds the horizontal sum of all elements in \a x. Equivalent to `reduce(x, std::plus<>{})`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE T hadd(const simd<T, Abi> &x) noexcept { return reduce(x, std::plus<>{}); }
		/** Finds the horizontal product of all elements in \a x. Equivalent to `reduce(x, std::multiplies<>{})`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE T hmul(const simd<T, Abi> &x) noexcept { return reduce(x, std::multiplies<>{}); }

		/** Finds the horizontal bitwise AND of all elements in \a x. Equivalent to `reduce(x, std::bit_and<>{})`. */
		template<std::integral T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE T hand(const simd<T, Abi> &x) noexcept { return reduce(x, std::bit_and<>{}); }
		/** Finds the horizontal bitwise XOR of all elements in \a x. Equivalent to `reduce(x, std::bit_xor<>{})`. */
		template<std::integral T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE T hxor(const simd<T, Abi> &x) noexcept { return reduce(x, std::bit_xor<>{}); }
		/** Finds the horizontal bitwise OR of all elements in \a x. Equivalent to `reduce(x, std::bit_or<>{})`. */
		template<std::integral T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE T hor(const simd<T, Abi> &x) noexcept { return reduce(x, std::bit_or<>{}); }
	}
#pragma endregion

#pragma region "simd casts"
	/** Implicitly converts elements of SIMD vector \a x to \a T or `T::value_type` if \a T is an instance of `simd`. */
	template<typename T, typename U, typename Abi>
	[[nodiscard]] constexpr auto simd_cast(const simd<U, Abi> &x) noexcept requires dpm::detail::valid_simd_cast<T, U, Abi>
	{
		detail::cast_return_t<T, U, Abi, simd<U, Abi>::size()> result = {};
		dpm::detail::copy_cast(x, result);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a x to \a T or `T::value_type` if \a T is an instance of `simd`. */
	template<typename T, typename U, typename Abi>
	[[nodiscard]] constexpr auto static_simd_cast(const simd<U, Abi> &x) noexcept requires dpm::detail::valid_simd_static_cast<T, U, Abi>
	{
		typename detail::static_cast_return<T, U, Abi, simd<U, Abi>::size()>::type result = {};
		dpm::detail::copy_cast(x, result);
		return result;
	}

	/** Converts SIMD vector \a x to it's fixed-size ABI equivalent for value type \a T. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr fixed_size_simd<T, simd_size_v<T, Abi>> to_fixed_size(const simd<T, Abi> &x) noexcept { return {x}; }
	/** Converts SIMD vector \a x to it's native ABI equivalent for value type \a T. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr native_simd<T> to_native(const simd<T, Abi> &x) noexcept { return {x}; }
	/** Converts SIMD vector \a x to it's compatible ABI equivalent for value type \a T. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T> to_compatible(const simd<T, Abi> &x) noexcept { return {x}; }
#pragma endregion

#pragma region "simd algorithms"
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> min(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::min(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> max(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::max(a[i], b[i]);
		return {result.data(), vector_aligned};
	}

	/** Returns an SIMD vector of minimum elements of \a a and scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> min(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::min(a[i], b);
		return {result.data(), vector_aligned};
	}
	/** Returns an SIMD vector of maximum elements of \a a and scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> max(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::max(a[i], b);
		return {result.data(), vector_aligned};
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr std::pair<simd<T, Abi>, simd<T, Abi>> minmax(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result[2] = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
		{
			const auto [min, max] = std::minmax(a[i], b[i]);
			result[0][i] = min;
			result[1][i] = max;
		}
		return {simd<T, Abi>{result[0].data(), vector_aligned}, simd<T, Abi>{result[1].data(), vector_aligned}};
	}
	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and scalar \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr std::pair<simd<T, Abi>, simd<T, Abi>> minmax(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result[2] = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
		{
			const auto [min, max] = std::minmax(a[i], b);
			result[0][i] = min;
			result[1][i] = max;
		}
		return {simd<T, Abi>{result[0].data(), vector_aligned}, simd<T, Abi>{result[1].data(), vector_aligned}};
	}

	/** Clamps elements of \a x between corresponding elements of \a min and \a max. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> clamp(const simd<T, Abi> &x, const simd<T, Abi> &min, const simd<T, Abi> &max) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::clamp(x[i], min[i], max[i]);
		return {result.data(), vector_aligned};
	}
	/** Clamps elements of \a x between \a min and \a max. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> clamp(const simd<T, Abi> &x, T min, T max) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::clamp(x[i], min, max);
		return {result.data(), vector_aligned};
	}
#pragma endregion
}
