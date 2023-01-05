/*
 * Created by switchblade on 2023-01-04.
 */

#pragma once

#include "../traits.hpp"

#ifndef SVM_USE_IMPORT

#include <utility>

#endif

namespace svm
{
	namespace detail
	{
		/* Clang-tidy was having issues with friended concept-constrained functions and types, so this is a workaround. */
		template<typename>
		struct simd_access;

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

	SVM_EXT_NAMESPACE_OPEN
	template<typename T, typename Abi, typename M>
	inline simd<T, Abi> blend(const simd<T, Abi> &a, const const_where_expression<M, simd<T, Abi>> &b);
	template<typename T, typename Abi, typename M>
	inline simd_mask<T, Abi> blend(const simd_mask<T, Abi> &a, const const_where_expression<M, simd_mask<T, Abi>> &b);

	template<typename T>
	inline T blend(const T &a, const const_where_expression<bool, T> &b);
	SVM_EXT_NAMESPACE_CLOSE

	/** @brief Type used to select element(s) of `T` using mask `M`.
	 * @tparam T SIMD vector, vector mask or scalar type, who's element(s) to select.
	 * @tparam M SIMD vector mask or `bool` used to select element(s) of `T`.
	 * @note If `M` is same as `bool`, `T` must be a scalar. */
	template<typename M, typename T> requires detail::valid_mask<M>::value && detail::valid_where_expression<M, T>
	class const_where_expression<M, T>
	{
		friend struct detail::simd_access<T>;
		friend struct detail::simd_access<M>;

		template<typename U, typename Abi, typename K>
		friend inline simd<U, Abi> SVM_EXT_NAMESPACE_IDENT(blend)(const simd<U, Abi> &, const const_where_expression<K, simd<U, Abi>> &);
		template<typename U, typename Abi, typename K>
		friend inline simd_mask<U, Abi> SVM_EXT_NAMESPACE_IDENT(blend)(const simd_mask<U, Abi> &, const const_where_expression<K, simd_mask<U, Abi>> &);

		template<typename U>
		friend inline U SVM_EXT_NAMESPACE_IDENT(blend)(const U &, const const_where_expression<bool, U> &);

	protected:
		using value_type = std::conditional_t<std::same_as<M, bool>, T, typename T::value_type>;

		template<typename U, typename Flags>
		constexpr static bool allow_copy_from = std::is_arithmetic_v<U> && is_simd_flag_type_v<Flags> && std::convertible_to<U, value_type>;
		template<typename U, typename Flags>
		constexpr static bool allow_copy_to = std::is_arithmetic_v<U> && is_simd_flag_type_v<Flags> && std::convertible_to<value_type, U>;

		constexpr static std::size_t data_size = std::same_as<M, bool> ? 1 : T::size();

	private:
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

	public:
		const_where_expression(const const_where_expression &) = delete;
		const_where_expression &operator=(const const_where_expression &) = delete;

		/** @warning Internal use only! */
		const_where_expression(M mask, T &data) noexcept : m_mask(mask), m_data(data) {}
		/** @warning Internal use only! */
		const_where_expression(M mask, const T &data) noexcept : m_mask(mask), m_data(const_cast<T &>(data)) {}

		[[nodiscard]] T operator-() const && noexcept requires (requires { -std::declval<T>(); })
		{
			T result;
			apply(result, [](auto, auto &&v, auto &&r) { return r = -v; });
			return result;
		}
		[[nodiscard]] T operator+() const && noexcept requires (requires { +std::declval<T>(); })
		{
			T result;
			apply(result, [](auto, auto &&v, auto &&r) { return r = +v; });
			return result;
		}
		[[nodiscard]] T operator~() const && noexcept requires (requires { ~std::declval<T>(); })
		{
			T result;
			apply(result, [](auto, auto &&v, auto &&r) { return r = ~v; });
			return result;
		}

		/** Copies selected elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const && noexcept requires allow_copy_to<U, Flags>
		{
			apply([&mem](std::size_t i, auto &&v) { mem[i] = static_cast<U>(v); });
		}

	protected:
		template<typename F>
		void apply(T &other, F &&f) const noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i)
				if (mask_at(m_mask, i)) f(i, data_at(m_data, i), data_at(other, i));
		}
		template<typename F>
		void apply(F &&f) const noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i)
				if (mask_at(m_mask, i)) f(i, data_at(m_data, i));
		}

		M m_mask;
		T &m_data;
	};

	/** @brief Type used to select elements of `T` using mask `M`. */
	template<typename M, typename T> requires detail::valid_mask<M>::value && detail::valid_where_expression<M, T>
	class where_expression<M, T> : public const_where_expression<M, T>
	{
		using value_type = typename const_where_expression<M, T>::value_type;

		template<typename U, typename Flags>
		constexpr static bool allow_copy_from = const_where_expression<M, T>::template allow_copy_from<U, Flags>;
		template<typename U, typename Flags>
		constexpr static bool allow_copy_to = const_where_expression<M, T>::template allow_copy_to<U, Flags>;

		using const_where_expression<M, T>::data_size;

		using const_where_expression<M, T>::apply;
		using const_where_expression<M, T>::m_mask;
		using const_where_expression<M, T>::m_data;

	public:
		using const_where_expression<M, T>::const_where_expression;

		template<std::convertible_to<T> U>
		void operator=(U &&value) && noexcept
		{
			const auto tmp = static_cast<T>(std::forward<U>(value));
			apply(tmp, [](auto, auto &&l, auto r) { l = r; });
		}

		void operator++() && noexcept requires (requires{ ++std::declval<T>(); }) { apply([](auto, auto &&v) { ++v; }); }
		void operator--() && noexcept requires (requires{ --std::declval<T>(); }) { apply([](auto, auto &&v) { --v; }); }
		void operator++(int) && noexcept requires (requires{ std::declval<T>()++; }) { apply([](auto, auto &&v) { v++; }); }
		void operator--(int) && noexcept requires (requires{ std::declval<T>()--; }) { apply([](auto, auto &&v) { v--; }); }

		template<typename U>
		void operator+=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a + b } -> std::convertible_to<T>; }
		{
			operator=(m_data + static_cast<T>(std::forward<U>(value)));
		}
		template<typename U>
		void operator-=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a - b } -> std::convertible_to<T>; }
		{
			operator=(m_data - static_cast<T>(std::forward<U>(value)));
		}

		template<typename U>
		void operator*=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a * b } -> std::convertible_to<T>; }
		{
			operator=(m_data * static_cast<T>(std::forward<U>(value)));
		}
		template<typename U>
		void operator/=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a / b } -> std::convertible_to<T>; }
		{
			operator=(m_data / static_cast<T>(std::forward<U>(value)));
		}
		template<typename U>
		void operator%=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a % b } -> std::convertible_to<T>; }
		{
			operator=(m_data % static_cast<T>(std::forward<U>(value)));
		}

		template<typename U>
		void operator&=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a & b } -> std::convertible_to<T>; }
		{
			operator=(m_data & static_cast<T>(std::forward<U>(value)));
		}
		template<typename U>
		void operator|=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a | b } -> std::convertible_to<T>; }
		{
			operator=(m_data | static_cast<T>(std::forward<U>(value)));
		}
		template<typename U>
		void operator^=(U &&value) && noexcept requires std::convertible_to<U, T> && requires(T a, T b) {{ a ^ b } -> std::convertible_to<T>; }
		{
			operator=(m_data ^ static_cast<T>(std::forward<U>(value)));
		}

		/** Copies selected elements from \a mem. */
		template<typename U, typename Flags>
		void copy_from(U *mem, Flags) const && noexcept requires allow_copy_from<U, Flags>
		{
			apply([&mem](std::size_t i, auto &&v) { v = static_cast<value_type>(mem[i]); });
		}
	};

	/** Selects elements of vector \a v using mask \a m. */
	template<typename T, typename Abi>
	inline where_expression<simd_mask<T, Abi>, simd<T, Abi>> where(const typename simd<T, Abi>::mask_type &m, simd<T, Abi> &v) noexcept
	{
		return {m, v};
	}
	/** Selects elements of vector \a v using mask \a m. */
	template<typename T, typename Abi>
	inline const_where_expression<simd_mask<T, Abi>, simd<T, Abi>> where(const typename simd<T, Abi>::mask_type &m, const simd<T, Abi> &v) noexcept
	{
		return {m, v};
	}
	/** Selects elements of mask \a v using mask \a m. */
	template<typename T, typename Abi>
	inline where_expression<simd_mask<T, Abi>, simd_mask<T, Abi>> where(const simd_mask<T, Abi> &m, simd_mask<T, Abi> &v) noexcept
	{
		return {m, v};
	}
	/** Selects elements of mask \a v using mask \a m. */
	template<typename T, typename Abi>
	inline const_where_expression<simd_mask<T, Abi>, simd_mask<T, Abi>> where(const simd_mask<T, Abi> &m, const simd_mask<T, Abi> &v) noexcept
	{
		return {m, v};
	}

	/** Selects \a v using value of \a m. */
	template<typename T>
	inline where_expression<bool, T> where(bool m, T &v) noexcept requires (!(is_simd_v<T> || is_simd_mask_v<T>)) { return {m, v}; }
	/** Selects \a v using value of \a m. */
	template<typename T>
	inline const_where_expression<bool, T> where(bool m, const T &v) noexcept requires (!(is_simd_v<T> || is_simd_mask_v<T>)) { return {m, v}; }

	/** Calculates a reduction of selected elements from \a value using \a binary_op and identity element \a identity. */
	template<typename M, typename V, typename Op>
	inline typename V::value_type reduce(const const_where_expression<M, V> &value, typename V::value_type identity, Op binary_op = {})
	{
		if (any_of(value.m_mask))
			return reduce(SVM_EXT_NAMESPACE_IDENT(blend)({identity}, value), binary_op);
		else
			return identity;
	}
	/** Calculates a reduction of selected element from \a value using \a binary_op and identity element \a identity. Equivalent to `binary_op(identity, +value)`. */
	template<typename V, typename Op>
	inline V reduce(const const_where_expression<bool, V> &value, V identity, Op binary_op = {}) { return std::invoke(binary_op, identity, +value); }

	SVM_EXT_NAMESPACE_OPEN
	/** Replaces elements of vector \a a with selected elements of where expression \a b. */
	template<typename T, typename Abi, typename M>
	[[nodiscard]] inline simd<T, Abi> blend(const simd<T, Abi> &a, const const_where_expression<M, simd<T, Abi>> &b)
	{
		simd<T, Abi> result = a;
		b.apply([&]<typename U>(std::size_t i, U &&val) { result[i] = std::forward<U>(val); });
		return result;
	}
	/** Replaces elements of mask \a a with selected elements of where expression \a b. */
	template<typename T, typename Abi, typename M>
	[[nodiscard]] inline simd_mask<T, Abi> blend(const simd_mask<T, Abi> &a, const const_where_expression<M, simd_mask<T, Abi>> &b)
	{
		simd_mask<T, Abi> result = a;
		b.apply([&](std::size_t i, auto &&val) { result[i] = val; });
		return result;
	}

	/** Returns either \a a or the selected element of where expression \a b. */
	template<typename T>
	[[nodiscard]] inline T blend(const T &a, const const_where_expression<bool, T> &b) { return b.m_mask ? b.m_value : a; }
	SVM_EXT_NAMESPACE_CLOSE
}