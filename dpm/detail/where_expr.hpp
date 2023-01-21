/*
 * Created by switchblade on 2023-01-04.
 */

#pragma once

#include "../traits.hpp"
#include "utility.hpp"

#ifndef DPM_USE_IMPORT

#include <utility>
#include <bit>

#endif

namespace dpm
{
	namespace detail
	{
		/* Clang-tidy was having issues with friended concept-constrained functions and types, so this is a workaround. */
		template<typename>
		struct native_access;

		template<typename>
		struct valid_mask : std::false_type {};
		template<>
		struct valid_mask<bool> : std::true_type {};
		template<typename T, typename Abi>
		struct valid_mask<simd_mask<T, Abi>> : std::true_type {};

		template<typename M, typename T>
		concept valid_where_expression = (requires { typename M::simd_type; } && std::same_as<typename M::simd_type, T>) ||
		                                 (std::same_as<M, bool> && std::is_arithmetic_v<T>) || std::same_as<M, T>;
	}

	template<typename T, typename M>
	class const_where_expression;
	template<typename T, typename M>
	class where_expression;

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, typename Abi, typename M>
		inline simd<T, Abi> blend(const simd<T, Abi> &a, const const_where_expression<M, simd<T, Abi>> &b);
		template<typename T, typename Abi, typename M>
		inline simd_mask<T, Abi> blend(const simd_mask<T, Abi> &a, const const_where_expression<M, simd_mask<T, Abi>> &b);

		template<typename T>
		inline T blend(const T &a, const const_where_expression<bool, T> &b);
	}

	/** @brief Type used to select element(s) of `T` using mask `M`.
	 * @tparam T SIMD vector, vector mask or scalar type, who's element(s) to select.
	 * @tparam M SIMD vector mask or `bool` used to select element(s) of `T`.
	 * @note If `M` is same as `bool`, `T` must be a scalar. */
	template<typename M, typename T>
	class const_where_expression
	{
		static_assert(detail::valid_mask<M>::value && detail::valid_where_expression<M, T>);

		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> ext::blend(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> ext::blend(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U ext::blend(const U &, const const_where_expression<bool, U> &);

	protected:
		using value_type = std::conditional_t<std::same_as<M, bool>, T, typename T::value_type>;

		[[nodiscard]] static decltype(auto) data_at(auto &data, std::size_t i) noexcept
		{
			if constexpr (!std::same_as<M, bool>)
				return data[i];
			else
				return data;
		}
		[[nodiscard]] static bool mask_at(auto &mask, std::size_t i) noexcept
		{
			if constexpr (!std::same_as<M, bool>)
				return mask[i];
			else
				return mask;
		}

		constexpr static std::size_t data_size = std::same_as<M, bool> ? 1 : T::size();

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		const_where_expression(M mask, T &data) noexcept : m_mask(mask), m_data(data) {}
		const_where_expression(M mask, const T &data) noexcept : m_mask(mask), m_data(const_cast<T &>(data)) {}

		[[nodiscard]] T operator-() const && noexcept requires (requires { -std::declval<T>(); })
		{
			return ext::blend(m_data, -m_data, m_mask);
		}
		[[nodiscard]] T operator+() const && noexcept requires (requires { +std::declval<T>(); })
		{
			return ext::blend(m_data, +m_data, m_mask);
		}
		[[nodiscard]] T operator~() const && noexcept requires (requires { ~std::declval<T>(); })
		{
			return ext::blend(m_data, ~m_data, m_mask);
		}

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const && noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < data_size; ++i) if (mask_at(m_mask, i)) mem[i] = static_cast<U>(data_at(m_data, i));
		}

	protected:
		M m_mask;
		T &m_data;
	};

	/** @brief Type used to select elements of `T` using mask `M`. */
	template<typename M, typename T>
	class where_expression : public const_where_expression<M, T>
	{
		using value_type = typename const_where_expression<M, T>::value_type;

		using const_where_expression<M, T>::mask_at;
		using const_where_expression<M, T>::data_at;
		using const_where_expression<M, T>::data_size;
		using const_where_expression<M, T>::m_mask;
		using const_where_expression<M, T>::m_data;

	public:
		using const_where_expression<M, T>::const_where_expression;

		template<std::convertible_to<T> U>
		void operator=(U &&value) && noexcept { m_data = ext::blend(m_data, static_cast<T>(std::forward<U>(value)), m_mask); }

		void operator++() && noexcept requires (requires{ ++std::declval<T>(); })
		{
			const auto old_data = m_data;
			const auto new_data = ++old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		void operator--() && noexcept requires (requires{ --std::declval<T>(); })
		{
			const auto old_data = m_data;
			const auto new_data = --old_data;
			m_data = ext::blend(old_data, new_data, m_mask);
		}
		void operator++(int) && noexcept requires (requires{ std::declval<T>()++; })
		{
			const auto old_data = m_data++;
			m_data = ext::blend(old_data, m_data, m_mask);
		}
		void operator--(int) && noexcept requires (requires{ std::declval<T>()--; })
		{
			const auto old_data = m_data--;
			m_data = ext::blend(old_data, m_data, m_mask);
		}

		template<typename U>
		void operator+=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a += b; }
		{
			auto new_data = m_data;
			new_data += static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator-=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a -= b; }
		{
			auto new_data = m_data;
			new_data -= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		void operator*=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a *= b; }
		{
			auto new_data = m_data;
			new_data *= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator/=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a /= b; }
		{
			auto new_data = m_data;
			new_data /= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator%=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a %= b; }
		{
			auto new_data = m_data;
			new_data %= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		template<typename U>
		void operator&=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a &= b; }
		{
			auto new_data = m_data;
			new_data &= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator|=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a |= b; }
		{
			auto new_data = m_data;
			new_data |= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}
		template<typename U>
		void operator^=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T &a, T b) { a ^= b; }
		{
			auto new_data = m_data;
			new_data ^= static_cast<T>(std::forward<U>(value));
			m_data = ext::blend(m_data, new_data, m_mask);
		}

		/** Copies selected elements from \a mem. */
		template<typename U, typename Flags>
		void copy_from(U *mem, Flags) && noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < data_size; ++i) if (mask_at(m_mask, i)) data_at(m_data, i) = static_cast<value_type>(mem[i]);
		}
	};

	/** Selects \a v using value of \a m. */
	template<typename T>
	inline where_expression<bool, T> where(bool m, T &v) noexcept requires (!(is_simd_v<T> || is_simd_mask_v<T>)) { return {m, v}; }
	/** Selects \a v using value of \a m. */
	template<typename T>
	inline const_where_expression<bool, T> where(bool m, const T &v) noexcept requires (!(is_simd_v<T> || is_simd_mask_v<T>)) { return {m, v}; }

	/** Calculates a reduction of selected elements from \a x using \a binary_op and identity element \a identity. */
	template<typename M, typename V, typename Op>
	inline typename V::value_type reduce(const const_where_expression<M, V> &x, typename V::value_type identity, Op binary_op = {})
	{
		if (any_of(x.m_mask))
			return reduce(ext::blend({identity}, x), binary_op);
		else
			return identity;
	}
	/** Calculates a sum of selected elements from \a x. Equivalent to `reduce(x, typename V::value_type{0}, binary_op)` */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type reduce(const const_where_expression<M, V> &x, std::plus<> binary_op) noexcept
	{
		return reduce(x, typename V::value_type{0}, binary_op);
	}
	/** Calculates a product of selected elements from \a x. Equivalent to `reduce(x, typename V::value_type{1}, binary_op)` */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type reduce(const const_where_expression<M, V> &x, std::multiplies<> binary_op) noexcept
	{
		return reduce(x, typename V::value_type{1}, binary_op);
	}
	/** Calculates a bitwise AND of selected elements from \a x. Equivalent to `reduce(x, typename V::value_type{ones-mask}, binary_op)` */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type reduce(const const_where_expression<M, V> &x, std::bit_and<> binary_op) noexcept
	{
		using mask_int = detail::uint_of_size_t<sizeof(typename V::value_type)>;
		return reduce(x, std::bit_cast<typename V::value_type>(~mask_int{0}), binary_op);
	}
	/** Calculates a bitwise OR of selected elements from \a x. Equivalent to `reduce(x, typename V::value_type{zeros-mask}, binary_op)` */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type reduce(const const_where_expression<M, V> &x, std::bit_or<> binary_op) noexcept
	{
		using mask_int = detail::uint_of_size_t<sizeof(typename V::value_type)>;
		return reduce(x, std::bit_cast<typename V::value_type>(mask_int{0}), binary_op);
	}
	/** Calculates a bitwise XOR of selected elements from \a x. Equivalent to `reduce(x, typename V::value_type{zeros-mask}, binary_op)` */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type reduce(const const_where_expression<M, V> &x, std::bit_xor<> binary_op) noexcept
	{
		using mask_int = detail::uint_of_size_t<sizeof(typename V::value_type)>;
		return reduce(x, std::bit_cast<typename V::value_type>(mask_int{0}), binary_op);
	}

	/** Finds the minimum of all selected elements (horizontal minimum) in \a x. */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type hmin(const const_where_expression<M, V> &x) noexcept
	{
		return reduce(x, std::numeric_limits<typename V::value_type>::max(), [](typename V::value_type a, typename V::value_type b) { return std::min(a, b); });
	}
	/** Finds the maximum of all selected elements (horizontal maximum) in \a x. */
	template<typename M, typename V>
	[[nodiscard]] inline typename V::value_type hmax(const const_where_expression<M, V> &x) noexcept
	{
		return reduce(x, std::numeric_limits<typename V::value_type>::min(), [](typename V::value_type a, typename V::value_type b) { return std::max(a, b); });
	}

	/** Calculates a reduction of selected element from \a x using \a binary_op and identity element \a identity. Equivalent to `binary_op(identity, +x)`. */
	template<typename V, typename Op>
	inline V reduce(const const_where_expression<bool, V> &x, V identity, Op binary_op = {}) { return std::invoke(binary_op, identity, +x); }

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Replaces elements of vector \a a with selected elements of where expression \a b. */
		template<typename T, typename Abi, typename M>
		[[nodiscard]] inline simd<T, Abi> blend(const simd<T, Abi> &a, const const_where_expression<M, simd<T, Abi>> &b)
		{
			return blend(a, b.m_data, b.m_mask);
		}
		/** Replaces elements of mask \a a with selected elements of where expression \a b. */
		template<typename T, typename Abi, typename M>
		[[nodiscard]] inline simd_mask<T, Abi> blend(const simd_mask<T, Abi> &a, const const_where_expression<M, simd_mask<T, Abi>> &b)
		{
			return blend(a, b.m_data, b.m_mask);
		}

		/** Returns either \a a or the selected element of where expression \a b. */
		template<typename T>
		[[nodiscard]] inline T blend(const T &a, const const_where_expression<bool, T> &b) { return blend(a, b.m_data, b.m_mask); }
	}
}