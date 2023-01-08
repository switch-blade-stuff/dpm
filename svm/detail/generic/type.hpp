/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../../flags.hpp"
#include "../where_expr.hpp"
#include "../reference.hpp"
#include "../assert.hpp"

#include "abi.hpp"

#ifndef SVM_USE_IMPORT

#include <span>

#endif

namespace svm
{
	template<typename T, typename Abi = simd_abi::compatible<T>>
	class simd_mask;

	template<typename T>
	using native_simd_mask = simd_mask<T, simd_abi::native<T>>;
	template<typename T, std::size_t N>
	using fixed_size_simd_mask = simd_mask<T, simd_abi::fixed_size<N>>;

	template<typename T, typename U, typename Abi>
	struct rebind_simd<T, simd_mask<U, Abi>> { using type = simd_mask<T, simd_abi::deduce_t<T, simd_size_v<U, Abi>, Abi>>; };
	template<std::size_t N, typename T, typename Abi>
	struct resize_simd<N, simd_mask<T, Abi>> { using type = simd_mask<T, simd_abi::deduce_t<T, N, Abi>>; };

	template<typename T, typename Abi = simd_abi::compatible<T>>
	class simd;

	template<typename T>
	using native_simd = simd_mask<T, simd_abi::native<T>>;
	template<typename T, std::size_t N>
	using fixed_size_simd = simd_mask<T, simd_abi::fixed_size<N>>;

	template<typename T, typename U, typename Abi>
	struct rebind_simd<T, simd<U, Abi>> { using type = simd<T, simd_abi::deduce_t<T, simd_size_v<U, Abi>, Abi>>; };
	template<std::size_t N, typename T, typename Abi>
	struct resize_simd<N, simd<T, Abi>> { using type = simd<T, simd_abi::deduce_t<T, N, Abi>>; };

	namespace detail
	{
		template<typename V, typename Abi>
		concept can_split_simd =is_simd_v<V> && !(simd_size_v<typename V::value_type, Abi> % V::size());
		template<typename V, typename Abi>
		concept can_split_mask = is_simd_mask_v<V> && !(simd_size_v<typename V::simd_type::value_type, Abi> % V::size());

		template<std::size_t N, std::size_t A>
		using avec = simd_abi::ext::aligned_vector<N, A>;

		struct bool_wrapper
		{
			bool_wrapper() = delete;

			constexpr bool_wrapper(bool value) noexcept : value(value) {}
			constexpr operator bool() const noexcept { return value; }

			bool value;
		};

		template<std::size_t N, std::size_t I = 0, typename G>
		inline SVM_SAFE_ARRAY void generate(auto &data, G &&gen) noexcept
		{
			if constexpr (I != N)
			{
				data[I] = std::invoke(gen, std::integral_constant<std::size_t, I>());
				generate<N, I + 1>(data, std::forward<G>(gen));
			}
		}

		template<typename From, typename To, typename FromAbi, typename ToAbi>
		inline SVM_SAFE_ARRAY void copy_cast(const simd_mask<From, FromAbi> &from, simd_mask<To, ToAbi> &to) noexcept
		{
			if constexpr (!std::same_as<simd_mask<From, FromAbi>, simd_mask<To, ToAbi>>)
			{
				constexpr auto result_align = std::max(alignof(simd_mask<From, FromAbi>), alignof(simd_mask<To, ToAbi>));
				constexpr auto result_size = std::max(simd_mask<From, FromAbi>::size(), simd_mask<To, ToAbi>::size());

				alignas(result_align) std::array<bool, result_size> tmp_buff;
				from.copy_to(tmp_buff.data(), vector_aligned);
				to.copy_from(tmp_buff.data(), vector_aligned);
			}
			else
				to = from;
		}
		template<typename From, typename To, typename FromAbi, typename ToAbi>
		inline SVM_SAFE_ARRAY void copy_cast(const simd<From, FromAbi> &from, simd<To, ToAbi> &to) noexcept
		{
			if constexpr (!std::same_as<simd<From, FromAbi>, simd<To, ToAbi>>)
			{
				constexpr auto result_align = std::max(alignof(simd<From, FromAbi>), alignof(simd<To, ToAbi>));
				constexpr auto result_size = std::max(simd<From, FromAbi>::size(), simd<To, ToAbi>::size());

				alignas(result_align) std::array<To, result_size> tmp_buff;
				from.copy_to(tmp_buff.data(), vector_aligned);
				to.copy_from(tmp_buff.data(), vector_aligned);
			}
			else
				to = from;
		}

		template<std::size_t I = 0, std::size_t N, typename T, typename Abi, typename... Abis>
		inline SVM_SAFE_ARRAY void concat_impl(std::array<bool, N> &buff, const simd_mask<T, Abi> &src, const simd_mask<T, Abis> &...other) noexcept
		{
			src.copy_to(buff.data() + I, vector_aligned);
			if constexpr (sizeof...(other) != 0) concat_impl<simd_mask<T, Abi>::size()>(buff, other...);
		}
		template<std::size_t I = 0, std::size_t N, typename T, typename Abi, typename... Abis>
		inline SVM_SAFE_ARRAY void concat_impl(std::array<T, N> &buff, const simd<T, Abi> &src, const simd<T, Abis> &...other) noexcept
		{
			src.copy_to(buff.data() + I, vector_aligned);
			if constexpr (sizeof...(other) != 0) concat_impl<simd_mask<T, Abi>::size()>(buff, other...);
		}
	}

	/** Returns \a value. */
	[[nodiscard]] constexpr bool all_of(detail::bool_wrapper value) noexcept { return value; }
	/** @copydoc any_of */
	[[nodiscard]] constexpr bool any_of(detail::bool_wrapper value) noexcept { return value; }
	/** Returns the negation of \a value. */
	[[nodiscard]] constexpr bool none_of(detail::bool_wrapper value) noexcept { return !value; }
	/** Returns `false`. */
	[[nodiscard]] constexpr bool some_of([[maybe_unused]] detail::bool_wrapper value) noexcept { return false; }
	/** Returns the integral representation of \a value. */
	[[nodiscard]] constexpr std::size_t popcount(detail::bool_wrapper value) noexcept { return static_cast<std::size_t>(value); }

	/** Returns `0`. */
	[[nodiscard]] constexpr std::size_t find_first_set([[maybe_unused]] detail::bool_wrapper value) noexcept { return 0; }
	/** @copydoc find_last_set */
	[[nodiscard]] constexpr std::size_t find_last_set([[maybe_unused]] detail::bool_wrapper value) noexcept { return 0; }

	SVM_DECLARE_EXT_NAMESPACE
	{
		/** Equivalent to `m ? b : a`. */
		template<typename T>
		[[nodiscard]] inline T blend(const T &a, const T &b, detail::bool_wrapper m) { return m ? b : a; }
	}

	/** @brief Type representing a data-parallel mask vector type.
	 * @tparam T Value type stored by the SIMD mask.
	 * @tparam Abi ABI used to select implementation of the SIMD mask. */
	template<typename T, typename Abi>
	class simd_mask
	{
	public:
		using value_type = bool;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::scalar;
		using simd_type = simd<T, abi_type>;

	public:
		/* The standard mandates the default specialization to have all constructors & assignment operators be deleted.
		 * See N4808 - 9.8.1 [parallel.simd.mask.overview] for details. */
		simd_mask() = delete;
		simd_mask(const simd_mask &) = delete;
		simd_mask &operator=(const simd_mask &) = delete;
		~simd_mask() = delete;
	};

	/** Preforms a bitwise AND on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator&(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] && b[i];
		return result;
	}
	/** Preforms a bitwise OR on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator|(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] || b[i];
		return result;
	}
	/** Preforms a bitwise XOR on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator^(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b[i];
		return result;
	}

	/** Preforms a bitwise AND on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd_mask<T, Abi> &operator&=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a & b; }
	/** Preforms a bitwise OR on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd_mask<T, Abi> &operator|=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a | b; }
	/** Preforms a bitwise XOR on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd_mask<T, Abi> &operator^=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a ^ b; }

	/** Preforms a logical AND on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator&&(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] && b[i];
		return result;
	}
	/** Preforms a logical OR on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator||(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] || b[i];
		return result;
	}

	/** Compares elements of masks \a a and \a b for equality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator==(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] == b[i];
		return result;
	}
	/** Compares elements of masks \a a and \a b for inequality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator!=(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] != b[i];
		return result;
	}

	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY bool all_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool result = true;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result &= mask[i];
		return result;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY bool any_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool result = false;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result |= mask[i];
		return result;
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY bool none_of(const simd_mask<T, Abi> &mask) noexcept { return !any_of(mask); }
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY bool some_of(const simd_mask<T, Abi> &mask) noexcept
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
	[[nodiscard]] inline SVM_SAFE_ARRAY std::size_t popcount(const simd_mask<T, Abi> &mask) noexcept
	{
		std::size_t result = 0;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result += mask[i];
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY std::size_t find_first_set(const simd_mask<T, Abi> &mask) noexcept
	{
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i) if (mask[i]) return i;
		return simd_mask<T, Abi>::size();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY std::size_t find_last_set(const simd_mask<T, Abi> &mask) noexcept
	{
		for (std::size_t i = simd_mask<T, Abi>::size(); i-- > 0;) if (mask[i]) return i;
		return simd_mask<T, Abi>::size();
	}

	/** Creates a where expression used to select elements of mask \a v using mask \a m. */
	template<typename T, typename Abi>
	inline where_expression<simd_mask<T, Abi>, simd_mask<T, Abi>> where(const simd_mask<T, Abi> &m, simd_mask<T, Abi> &v) noexcept
	{
		return {m, v};
	}
	/** Creates a where expression used to select elements of mask \a v using mask \a m. */
	template<typename T, typename Abi>
	inline const_where_expression<simd_mask<T, Abi>, simd_mask<T, Abi>> where(const simd_mask<T, Abi> &m, const simd_mask<T, Abi> &v) noexcept
	{
		return {m, v};
	}

	/** Converts SIMD mask \a value to it's fixed-size equivalent for value type `T`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY fixed_size_simd_mask<T, simd_size_v<T, Abi>> to_fixed_size(const simd_mask<T, Abi> &value) noexcept
	{
		fixed_size_simd_mask<T, simd_size_v<T, Abi>> result;
		detail::copy_cast(value, result);
		return result;
	}
	/** Converts SIMD mask \a value to it's native ABI equivalent for value type `T`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY native_simd_mask<T> to_native(const simd_mask<T, Abi> &value) noexcept
	{
		native_simd_mask<T> result;
		detail::copy_cast(value, result);
		return result;
	}
	/** Converts SIMD mask \a value to it's compatible ABI equivalent for value type `T`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T> to_compatible(const simd_mask<T, Abi> &value) noexcept
	{
		simd_mask<T> result;
		detail::copy_cast(value, result);
		return result;
	}

	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask a copy of the `i + j * V::size()`th element from \a value.
	 * @note Size of \a value must be a multiple of `V::size()`. */
	template<typename V, typename Abi, typename U = typename V::simd_type::value_type>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto split(const simd_mask<typename V::simd_type::value_type, Abi> &value) noexcept requires detail::can_split_mask<V, Abi>
	{
		std::array<V, simd_size_v<U, Abi> / V::size()> result;
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			for (std::size_t i = 0; i < V::size(); ++i)
				result[j][i] = value[j * V::size() + i];
		}
		return result;
	}
	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a value.
	 * @note `N` must be a multiple of `simd_size_v<T, Abi>::size()`. */
	template<std::size_t N, typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto split_by(const simd_mask<T, Abi> &value) noexcept requires (simd_size_v<T, Abi> % N == 0)
	{
		constexpr auto split_size = simd_size_v<T, Abi> / N;
		std::array<resize_simd_t<split_size, simd_mask<T, Abi>>, N> result;
		for (std::size_t j = 0; j < N; ++j)
		{
			for (std::size_t i = 0; i < split_size; ++i)
				result[j][i] = value[j * split_size + i];
		}
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, typename... Abis>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto concat(const simd_mask<T, Abis> &...values) noexcept
	{
		using result_t = simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>>;
		alignas(std::max({alignof(result_t), alignof(simd_mask<T, Abis>)...})) std::array<bool, result_t::size()> tmp_buff;
		result_t result;

		detail::concat_impl(tmp_buff, result, values...);
		result.copy_from(tmp_buff.data(), vector_aligned);
		return result;
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, typename Abi, std::size_t N>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto concat(const std::array<simd_mask<T, Abi>, N> &values) noexcept
	{
		using result_t = resize_simd_t<simd_size_v<T, Abi> * N, simd_mask<T, Abi>>;
		alignas(std::max(alignof(result_t), alignof(simd_mask<T, Abi>))) std::array<bool, result_t::size()> tmp_buff;
		result_t result;

		for (std::size_t i = 0, j = 0; i < tmp_buff.size(); i += simd_size_v<T, Abi>, ++j)
			values[j].copy_to(tmp_buff.data() + i, vector_aligned);
		result.copy_from(tmp_buff.data(), vector_aligned);
		return result;
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		template<detail::vectorizable T>
		struct native_data_type<simd_mask<T, simd_abi::scalar>> { using type = bool; };
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		struct native_data_type<simd_mask<T, detail::avec<N, Align>>> { using type = bool; };

		template<detail::vectorizable T>
		struct native_data_size<simd_mask<T, simd_abi::scalar>> : std::integral_constant<std::size_t, 1> {};
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		struct native_data_size<simd_mask<T, detail::avec<N, Align>>> : std::integral_constant<std::size_t, N> {};

		/** Replaces elements of masks \a and \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, typename Abi, typename M>
		[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> blend(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b, const simd_mask<T, Abi> &m)
		{
			simd_mask<T, Abi> result = a;
			for (std::size_t i = 0; i < m.size(); ++i) result[i] = m[i] ? b[i] : a[i];
			return result;
		}

		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<bool, 1> to_native_data(simd_mask<T, simd_abi::scalar> &value) noexcept;
		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<const bool, 1> to_native_data(const simd_mask<T, simd_abi::scalar> &value) noexcept;

		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		[[nodiscard]] inline std::span<T, N> to_native_data(simd_mask<T, detail::avec<N, Align>> &value) noexcept;
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		[[nodiscard]] inline std::span<const T, N> to_native_data(const simd_mask<T, detail::avec<N, Align>> &value) noexcept;
	}

	template<detail::vectorizable T>
	class simd_mask<T, simd_abi::scalar>
	{
		friend struct detail::simd_access<simd_mask>;

	public:
		using value_type = bool;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::scalar;
		using simd_type = simd<T, abi_type>;

		/** Returns width of the SIMD mask (always 1 for `simd_abi::scalar`). */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the underlying boolean with \a value. */
		constexpr simd_mask(value_type value) noexcept : m_value(value) {}
		/** Initializes the underlying boolean from the value pointed to by \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying boolean from the value pointed to by \a mem. */
		template<typename Flags>
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { m_value = mem[0]; }
		/** Copies the underlying boolean to the value pointed to by \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { mem[0] = m_value; }

		[[nodiscard]] reference operator[]([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size());
			return reference{m_value};
		}
		[[nodiscard]] value_type operator[]([[maybe_unused]] std::size_t i) const noexcept
		{
			SVM_ASSERT(i < size());
			return m_value;
		}

		[[nodiscard]] simd_mask operator!() const noexcept { return {!m_value}; }

	private:
		value_type m_value;
	};

	template<detail::vectorizable T, std::size_t N, std::size_t Align>
	class simd_mask<T, detail::avec<N, Align>>
	{
		friend struct detail::simd_access<simd_mask>;

	public:
		using value_type = bool;
		using reference = detail::simd_reference<value_type>;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<T, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		constexpr static std::size_t alignment = std::max(Align, alignof(bool[N]));

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the underlying elements with \a value. */
		constexpr SVM_SAFE_ARRAY simd_mask(value_type value) noexcept { std::fill_n(m_data, size(), value); }
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		SVM_SAFE_ARRAY simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (alignment != alignof(value_type))
				other.copy_to(m_data, overaligned<alignment>);
			else
				other.copy_to(m_data, element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		SVM_SAFE_ARRAY simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		SVM_SAFE_ARRAY void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { std::copy_n(mem, size(), m_data); }
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		SVM_SAFE_ARRAY void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { std::copy_n(m_data, size(), mem); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			SVM_ASSERT(i < size());
			return reference{m_data[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			SVM_ASSERT(i < size());
			return m_data[i];
		}

		[[nodiscard]] SVM_SAFE_ARRAY simd_mask operator!() const noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < size(); ++i)
				result[i] = !m_data[i];
			return result;
		}

	private:
		alignas(alignment) value_type m_data[size()];
	};

	namespace detail
	{
		template<detail::vectorizable T>
		struct simd_access<simd_mask<T, simd_abi::scalar>>
		{
			using mask_t = simd_mask<T, simd_abi::scalar>;

			[[nodiscard]] static std::span<bool, 1> to_native_data(mask_t &value) noexcept { return std::span<bool, 1>{&value.m_value, &value.m_value + 1}; }
			[[nodiscard]] static std::span<const bool, 1> to_native_data(const mask_t &value) noexcept { return std::span<const bool, 1>{&value.m_value, &value.m_value + 1}; }
		};
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		struct simd_access<simd_mask<T, detail::avec<N, Align>>>
		{
			using mask_t = simd_mask<T, detail::avec<N, Align>>;

			[[nodiscard]] static std::span<bool, N> to_native_data(mask_t &value) noexcept { return {value.m_data}; }
			[[nodiscard]] static std::span<const bool, N> to_native_data(const mask_t &value) noexcept { return {value.m_data}; }
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a (single-element) span of the underlying boolean of \a value. */
		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<bool, 1> to_native_data(simd_mask<T, simd_abi::scalar> &value) noexcept
		{
			return detail::simd_access<simd_mask<T, simd_abi::scalar>>::to_native_data(value);
		}
		/** Returns a constant (single-element) span of the underlying boolean of \a value. */
		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<const bool, 1> to_native_data(const simd_mask<T, simd_abi::scalar> &value) noexcept
		{
			return detail::simd_access<simd_mask<T, simd_abi::scalar>>::to_native_data(value);
		}

		/** Returns a span of the underlying booleans of \a value. */
		template<detail::vectorizable T, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<T, N> to_native_data(simd_mask<T, detail::avec<N, A>> &value) noexcept
		{
			return detail::simd_access<simd_mask<T, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a constant span of the underlying booleans of \a value. */
		template<detail::vectorizable T, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const T, N> to_native_data(const simd_mask<T, detail::avec<N, A>> &value) noexcept
		{
			return detail::simd_access<simd_mask<T, detail::avec<N, A>>>::to_native_data(value);
		}
	}

	/** @brief Type representing a data-parallel arithmetic vector.
	 * @tparam T Value type stored by the SIMD vector.
	 * @tparam Abi ABI used to select implementation of the SIMD vector. */
	template<typename T, typename Abi>
	class simd
	{
	public:
		using value_type = T;
		using reference = detail::simd_reference<value_type>;

		using abi_type = Abi;
		using mask_type = simd_mask<T, abi_type>;

	public:
		/* The standard mandates the default specialization to have all constructors & assignment operators be deleted.
		 * See N4808 - 9.6.1 [parallel.simd.overview] for details. */
		simd() = delete;
		simd(const simd &) = delete;
		simd &operator=(const simd &) = delete;
		~simd() = delete;
	};

	/** Adds elements of vector \a b to elements of vector \a a, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator+(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l + r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] + b[i];
		return result;
	}
	/** Subtracts elements of vector \a b from elements of vector \a a, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator-(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l - r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] - b[i];
		return result;
	}

	/** Adds elements of vector \a b to elements of vector \a a, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator+=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l += r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] += b[i];
		return a;
	}
	/** Subtracts elements of vector \a b from elements of vector \a a, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator-=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l -= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] -= b[i];
		return a;
	}

	/** Multiplies elements of vector \a a by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator*(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l * r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] * b[i];
		return result;
	}
	/** Divides elements of vector \a a by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator/(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l / r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] / b[i];
		return result;
	}
	/** Preforms a modulo operation of elements of vector \a a by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator%(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l % r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] % b[i];
		return result;
	}

	/** Multiplies elements of vector \a a by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator*=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l *= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] *= b[i];
		return a;
	}
	/** Divides elements of vector \a a by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator/=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l /= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] /= b[i];
		return a;
	}
	/** Preforms a modulo operation of elements of vector \a a by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator%=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l %= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] %= b[i];
		return a;
	}

	/** Preforms a bitwise AND between elements of vector \a a and elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator&(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l & r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] & b[i];
		return result;
	}
	/** Preforms a bitwise OR between elements of vector \a a and elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator|(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l | r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] | b[i];
		return result;
	}
	/** Preforms a bitwise XOR between elements of vector \a a and elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator^(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l ^ r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b[i];
		return result;
	}

	/** Preforms a bitwise AND between elements of vector \a a and elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator&=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l &= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] &= b[i];
		return a;
	}
	/** Preforms a bitwise OR between elements of vector \a a and elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator|=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l |= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] |= b[i];
		return a;
	}
	/** Preforms a bitwise XOR between elements of vector \a a and elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator^=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l ^= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] ^= b[i];
		return a;
	}

	/** Shifts elements of vector \a a left by the amount specified by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator<<(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l << r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] << b[i];
		return result;
	}
	/** Shifts elements of vector \a a right by the amount specified by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator>>(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l >> r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >> b[i];
		return result;
	}

	/** Shifts elements of vector \a a left by the amount specified by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator<<=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l <<= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] <<= b[i];
		return a;
	}
	/** Shifts elements of vector \a a right by the amount specified by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator>>=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l >>= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] >>= b[i];
		return a;
	}

	/** Shifts elements of vector \a a left by \a n, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator<<(const simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l << r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] << n;
		return result;
	}
	/** Shifts elements of vector \a a rught by \a n, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> operator>>(const simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l >> r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >> n;
		return result;
	}

	/** Shifts elements of vector \a a left by \a n, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator<<=(simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l <<= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] <<= n;
		return a;
	}
	/** Shifts elements of vector \a a right by \a n, and returns reference to \a. */
	template<typename T, typename Abi>
	inline SVM_SAFE_ARRAY simd<T, Abi> &operator>>=(simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l >>= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] >>= n;
		return a;
	}

	/** Compares elements of vectors \a a and \a b for equality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator==(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] == b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for equality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator!=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] != b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for less-than or equal, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator<=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] <= b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for greater-than or equal, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator>=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >= b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for less-than, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator<(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] < b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for greater-than, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd_mask<T, Abi> operator>(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] > b[i];
		return result;
	}

	namespace detail
	{
		template<typename T, typename Op>
		[[nodiscard]] inline T reduce_pair(T a, T b, Op binary_op)
		{
			if constexpr (!std::is_invocable_r_v<T, Op, T, T>)
				return std::invoke(binary_op, simd<T, simd_abi::scalar>{a}, simd<T, simd_abi::scalar>{b})[0];
			else
				return std::invoke(binary_op, a, b);
		}
		template<typename T, std::size_t N, typename Op>
		[[nodiscard]] inline SVM_SAFE_ARRAY T reduce_array(const std::array<T, N> &data, Op binary_op)
		{
			if constexpr (N == 1)
				return data[0];
			else if constexpr (N == 2)
				return reduce_pair(data[0], data[1], binary_op);
			else
			{
				std::array<T, N - N / 2> b;
				std::array<T, N / 2> a;

				/* Separate `value` into halves and reduce them separately. */
				for (std::size_t i = 0; i < a.size(); ++i)
					a[i] = data[0];
				for (std::size_t i = 0, j = a.size(); i < b.size(); ++i, ++j)
					b[i] = data[j];

				return reduce_pair(reduce_array(a, binary_op), reduce_array(b, binary_op), binary_op);
			}
		}
		template<std::size_t N, typename T, typename Abi, typename Op>
		[[nodiscard]] inline SVM_SAFE_ARRAY T reduce_impl(const simd<T, Abi> &value, Op binary_op)
		{
			alignas(simd<T, Abi>) std::array<T, simd<T, Abi>::size()> buff;
			value.copy_to(buff.data(), vector_aligned);
			return reduce_array(buff, binary_op);
		}
	}

	/** Creates a where expression used to select elements of vector \a v using mask \a m. */
	template<typename T, typename Abi>
	inline where_expression<simd_mask<T, Abi>, simd<T, Abi>> where(const typename simd<T, Abi>::mask_type &m, simd<T, Abi> &v) noexcept
	{
		return {m, v};
	}
	/** Creates a where expression used to select elements of vector \a v using mask \a m. */
	template<typename T, typename Abi>
	inline const_where_expression<simd_mask<T, Abi>, simd<T, Abi>> where(const typename simd<T, Abi>::mask_type &m, const simd<T, Abi> &v) noexcept
	{
		return {m, v};
	}

	/** Calculates a reduction of all elements from \a value using \a binary_op. */
	template<typename T, typename Abi, typename Op = std::plus<>>
	inline T reduce(const simd<T, Abi> &value, Op binary_op = {}) { return detail::reduce_impl<simd_size_v<T, Abi>>(value, binary_op); }

	/** Finds the minimum of all elements (horizontal minimum) in \a value. */
	template<typename T, typename Abi>
	[[nodiscard]] inline T hmin(const simd<T, Abi> &value) noexcept { return reduce(value, [](T a, T b) { return std::min(a, b); }); }
	/** Finds the maximum of all elements (horizontal maximum) in \a value. */
	template<typename T, typename Abi>
	[[nodiscard]] inline T hmax(const simd<T, Abi> &value) noexcept { return reduce(value, [](T a, T b) { return std::max(a, b); }); }

	namespace detail
	{
		template<typename T>
		struct deduce_cast { using type = T; };
		template<typename T> requires is_simd_v<T>
		struct deduce_cast<T> { using type = typename T::value_type; };

		template<typename T, typename U, typename Abi, std::size_t N>
		struct cast_return { using type = std::conditional_t<std::same_as<T, U>, simd<T, Abi>, simd<T, simd_abi::deduce_t<T, N, Abi, simd_abi::fixed_size<N>>>>; };
		template<typename T, typename U, typename Abi, std::size_t N> requires is_simd_v<T>
		struct cast_return<T, U, Abi, N> { using type = T; };

		template<typename T, typename U, typename Abi, std::size_t N>
		struct static_cast_return { using type = std::conditional_t<std::same_as<T, U>, simd<T, Abi>, simd<T, simd_abi::deduce_t<T, N, Abi, simd_abi::fixed_size<N>>>>; };
		template<std::integral T, std::integral U, typename Abi, std::size_t N>
		struct static_cast_return<T, U, Abi, N> { using type = std::conditional_t<std::same_as<T, U> || std::same_as<std::make_signed_t<T>, std::make_signed_t<U>>, simd<T, Abi>, simd<T, simd_abi::deduce_t<T, N, Abi, simd_abi::fixed_size<N>>>>; };
		template<typename T, typename U, typename Abi, std::size_t N> requires is_simd_v<T>
		struct static_cast_return<T, U, Abi, N> { using type = T; };

		template<typename T, typename U, typename Abi>
		struct equal_cast_size : std::bool_constant<simd<U, Abi>::size() == T::size()> {};
		template<typename T, typename U, typename Abi> requires (!is_simd_v<T>)
		struct equal_cast_size<T, U, Abi> : std::true_type {};
	}

	/** Implicitly converts elements of SIMD vector \a value to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, typename Abi, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto simd_cast(const simd<U, Abi> &value) noexcept requires (std::is_convertible_v<U, To> && detail::equal_cast_size<T, U, Abi>::value)
	{
		typename detail::cast_return<T, U, Abi, simd<U, Abi>::size()>::type result;
		detail::copy_cast(value, result);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a value to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, typename Abi, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto static_simd_cast(const simd<U, Abi> &value) noexcept requires (std::convertible_to<U, To> && detail::equal_cast_size<T, U, Abi>::value)
	{
		typename detail::static_cast_return<T, U, Abi, simd<U, Abi>::size()>::type result;
		detail::copy_cast(value, result);
		return result;
	}

	/** Converts SIMD vector \a value to it's fixed-size ABI equivalent for value type `T`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY fixed_size_simd<T, simd_size_v<T, Abi>> to_fixed_size(const simd<T, Abi> &value) noexcept
	{
		fixed_size_simd<T, simd_size_v<T, Abi>> result;
		detail::copy_cast(value, result);
		return result;
	}
	/** Converts SIMD vector \a value to it's native ABI equivalent for value type `T`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY native_simd<T> to_native(const simd<T, Abi> &value) noexcept
	{
		native_simd<T> result;
		detail::copy_cast(value, result);
		return result;
	}
	/** Converts SIMD vector \a value to it's compatible ABI equivalent for value type `T`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY simd<T> to_compatible(const simd<T, Abi> &value) noexcept
	{
		simd<T> result;
		detail::copy_cast(value, result);
		return result;
	}

	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * V::size()`th element from \a value.
	 * @note Size of \a value must be a multiple of `V::size()`. */
	template<typename V, typename Abi, typename U = typename V::value_type>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto split(const simd<U, Abi> &value) noexcept requires detail::can_split_simd<V, Abi>
	{
		alignas(std::max(alignof(V), alignof(simd<U, Abi>))) std::array<U, simd<U, Abi>::size()> tmp_buff;
		std::array<V, simd_size_v<U, Abi> / V::size()> result;

		value.copy_to(tmp_buff.data(), vector_aligned);
		for (std::size_t i = 0, j = 0; i < result.size(); ++i, j += V::size())
			result[j].copy_from(tmp_buff.data() + j, vector_aligned);
		return result;
	}
	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a value.
	 * @note `N` must be a multiple of `simd_size_v<T, Abi>::size()`. */
	template<std::size_t N, typename T, typename Abi>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto split_by(const simd<T, Abi> &value) noexcept requires (simd_size_v<T, Abi> % N == 0)
	{
		constexpr auto split_size = simd_size_v<T, Abi> / N;
		using split_type = resize_simd_t<split_size, simd<T, Abi>>;

		alignas(std::max(alignof(split_type), alignof(simd<T, Abi>))) std::array<T, simd<T, Abi>::size()> tmp_buff;
		std::array<split_type, N> result;

		value.copy_to(tmp_buff.data(), vector_aligned);
		for (std::size_t i = 0, j = 0; i < result.size(); ++i, j += split_size)
			result[i].copy_from(tmp_buff.data() + j, vector_aligned);
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD vector. */
	template<typename T, typename... Abis>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto concat(const simd<T, Abis> &...values) noexcept
	{
		using result_t = simd<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>>;
		alignas(std::max({alignof(result_t), alignof(simd<T, Abis>)...})) std::array<T, result_t::size()> tmp_buff;
		result_t result;

		detail::concat_impl(tmp_buff, result, values...);
		result.copy_from(tmp_buff.data(), vector_aligned);
		return result;
	}
	/** Concatenates elements of \a values into a single SIMD vector. */
	template<typename T, typename Abi, std::size_t N>
	[[nodiscard]] inline SVM_SAFE_ARRAY auto concat(const std::array<simd<T, Abi>, N> &values) noexcept
	{
		using result_t = resize_simd_t<simd_size_v<T, Abi> * N, simd<T, Abi>>;
		alignas(std::max(alignof(result_t), alignof(simd<T, Abi>))) std::array<T, result_t::size()> tmp_buff;
		result_t result;

		for (std::size_t i = 0, j = 0; i < tmp_buff.size(); i += simd_size_v<T, Abi>, ++j)
			values[j].copy_to(tmp_buff.data() + i, vector_aligned);
		result.copy_from(tmp_buff.data(), vector_aligned);
		return result;
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		template<detail::vectorizable T>
		struct native_data_type<simd<T, simd_abi::scalar>> { using type = T; };
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		struct native_data_type<simd<T, detail::avec<N, Align>>> { using type = T; };

		template<detail::vectorizable T>
		struct native_data_size<simd<T, simd_abi::scalar>> : std::integral_constant<std::size_t, 1> {};
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		struct native_data_size<simd<T, detail::avec<N, Align>>> : std::integral_constant<std::size_t, N> {};

		/** Replaces elements of vectors \a and \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline SVM_SAFE_ARRAY simd<T, Abi> blend(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd_mask<T, Abi> &m)
		{
			simd<T, Abi> result = a;
			for (std::size_t i = 0; i < m.size(); ++i) result[i] = m[i] ? b[i] : a[i];
			return result;
		}

		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<T, 1> to_native_data(simd<T, simd_abi::scalar> &value) noexcept;
		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<const T, 1> to_native_data(const simd<T, simd_abi::scalar> &value) noexcept;

		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		[[nodiscard]] inline std::span<T, N> to_native_data(simd<T, detail::avec<N, Align>> &value) noexcept;
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		[[nodiscard]] inline std::span<const T, N> to_native_data(const simd<T, detail::avec<N, Align>> &value) noexcept;
	}

	template<detail::vectorizable T>
	class simd<T, simd_abi::scalar>
	{
		friend struct detail::simd_access<simd>;

	public:
		using value_type = T;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::scalar;
		using mask_type = simd_mask<T, abi_type>;

		/** Returns width of the SIMD vector (always 1 for `simd_abi::scalar`). */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the underlying scalar with \a value. */
		template<detail::compatible_element<value_type> U>
		constexpr simd(U &&value) noexcept : m_value(static_cast<value_type>(value)) {}
		/** Initializes the underlying scalar with a value provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept : simd(std::invoke(gen, std::integral_constant<std::size_t, 0>())) {}
		/** Initializes the underlying scalar from the value pointed to by \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying scalar from the value pointed to by \a mem. */
		template<typename U, typename Flags>
		void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { m_value = static_cast<value_type>(mem[0]); }
		/** Copies the underlying scalar to the value pointed to by \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { mem[0] = static_cast<U>(m_value); }

		[[nodiscard]] reference operator[]([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size());
			return reference{m_value};
		}
		[[nodiscard]] value_type operator[]([[maybe_unused]] std::size_t i) const noexcept
		{
			SVM_ASSERT(i < size());
			return m_value;
		}

		simd operator++(int) noexcept requires (requires { std::declval<value_type &>()++; }) { return {m_value++}; }
		simd operator--(int) noexcept requires (requires { std::declval<value_type &>()--; }) { return {m_value--}; }
		simd &operator++() noexcept requires (requires { ++std::declval<value_type &>(); })
		{
			++m_value;
			return *this;
		}
		simd &operator--() noexcept requires (requires { --std::declval<value_type &>(); })
		{
			--m_value;
			return *this;
		}

		[[nodiscard]] mask_type operator!() const noexcept requires (requires { !std::declval<value_type &>(); }) { return {!m_value}; }
		[[nodiscard]] simd operator~() const noexcept requires (requires { ~std::declval<value_type &>(); }) { return {~m_value}; }
		[[nodiscard]] simd operator+() const noexcept requires (requires { +std::declval<value_type &>(); }) { return {+m_value}; }
		[[nodiscard]] simd operator-() const noexcept requires (requires { -std::declval<value_type &>(); }) { return {-m_value}; }

	private:
		value_type m_value;
	};

	template<detail::vectorizable T, std::size_t N, std::size_t Align>
	class simd<T, detail::avec<N, Align>>
	{
		friend struct detail::simd_access<simd>;

	public:
		using value_type = T;
		using reference = detail::simd_reference<value_type>;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<T, abi_type>;

		/** Returns width of the SIMD vector. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		constexpr static std::size_t alignment = std::max(Align, alignof(value_type[N]));

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		constexpr SVM_SAFE_ARRAY simd(U &&value) noexcept { std::fill_n(m_data, size(), static_cast<value_type>(std::forward<U>(value))); }
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		SVM_SAFE_ARRAY simd(G &&gen) noexcept { detail::generate<size()>(m_data, std::forward<G>(gen)); }

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		SVM_SAFE_ARRAY simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		SVM_SAFE_ARRAY simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		SVM_SAFE_ARRAY void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { std::copy_n(mem, size(), m_data); }
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		SVM_SAFE_ARRAY void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { std::copy_n(m_data, size(), mem); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			SVM_ASSERT(i < size());
			return reference{m_data[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			SVM_ASSERT(i < size());
			return m_data[i];
		}

		SVM_SAFE_ARRAY simd operator++(int) noexcept requires (requires { std::declval<value_type &>()++; })
		{
			simd result;
			for (std::size_t i = 0; i < size(); ++i) result[i] = m_data[i]++;
			return result;
		}
		SVM_SAFE_ARRAY simd operator--(int) noexcept requires (requires { std::declval<value_type &>()--; })
		{
			simd result;
			for (std::size_t i = 0; i < size(); ++i) result[i] = m_data[i]--;
			return result;
		}
		SVM_SAFE_ARRAY simd &operator++() noexcept requires (requires { ++std::declval<value_type &>(); })
		{
			for (std::size_t i = 0; i < size(); ++i) ++m_data[i];
			return *this;
		}
		SVM_SAFE_ARRAY simd &operator--() noexcept requires (requires { --std::declval<value_type &>(); })
		{
			for (std::size_t i = 0; i < size(); ++i) --m_data[i];
			return *this;
		}

		[[nodiscard]] SVM_SAFE_ARRAY mask_type operator!() const noexcept requires (requires { !std::declval<value_type &>(); })
		{
			mask_type result;
			for (std::size_t i = 0; i < size(); ++i) result[i] = !m_data[i];
			return result;
		}
		[[nodiscard]] SVM_SAFE_ARRAY simd operator~() const noexcept requires (requires { ~std::declval<value_type &>(); })
		{
			simd result;
			for (std::size_t i = 0; i < size(); ++i) result[i] = ~m_data[i];
			return result;
		}
		[[nodiscard]] SVM_SAFE_ARRAY simd operator+() const noexcept requires (requires { +std::declval<value_type &>(); }) { return *this; }
		[[nodiscard]] SVM_SAFE_ARRAY simd operator-() const noexcept requires (requires { -std::declval<value_type &>(); })
		{
			simd result;
			for (std::size_t i = 0; i < size(); ++i) result[i] = -m_data[i];
			return result;
		}

	private:
		alignas(alignment) value_type m_data[size()];
	};

	namespace detail
	{
		template<detail::vectorizable T>
		struct simd_access<simd<T, simd_abi::scalar>>
		{
			using simd_t = simd<T, simd_abi::scalar>;

			[[nodiscard]] static std::span<T, 1> to_native_data(simd_t &value) noexcept { return std::span<T, 1>{&value.m_value, &value.m_value + 1}; }
			[[nodiscard]] static std::span<const T, 1> to_native_data(const simd_t &value) noexcept { return std::span<const T, 1>{&value.m_value, &value.m_value + 1}; }
		};
		template<detail::vectorizable T, std::size_t N, std::size_t Align>
		struct simd_access<simd<T, detail::avec<N, Align>>>
		{
			using simd_t = simd<T, detail::avec<N, Align>>;

			[[nodiscard]] static std::span<T, N> to_native_data(simd_t &value) noexcept { return {value.m_data}; }
			[[nodiscard]] static std::span<const T, N> to_native_data(const simd_t &value) noexcept { return {value.m_data}; }
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a (single-element) span of the underlying scalar of \a value. */
		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<T, 1> to_native_data(simd<T, simd_abi::scalar> &value) noexcept
		{
			return detail::simd_access<simd<T, simd_abi::scalar>>::to_native_data(value);
		}
		/** Returns a constant (single-element) span of the underlying scalar of \a value. */
		template<detail::vectorizable T>
		[[nodiscard]] inline std::span<const T, 1> to_native_data(const simd<T, simd_abi::scalar> &value) noexcept
		{
			return detail::simd_access<simd<T, simd_abi::scalar>>::to_native_data(value);
		}

		/** Returns a span of the underlying elements of \a value. */
		template<detail::vectorizable T, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<T, N> to_native_data(simd<T, detail::avec<N, A>> &value) noexcept
		{
			return detail::simd_access<simd<T, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a constant span of the underlying elements of \a value. */
		template<detail::vectorizable T, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const T, N> to_native_data(const simd<T, detail::avec<N, A>> &value) noexcept
		{
			return detail::simd_access<simd<T, detail::avec<N, A>>>::to_native_data(value);
		}
	}
}