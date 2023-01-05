/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../../flags.hpp"
#include "../where_expr.hpp"
#include "../reference.hpp"
#include "../assert.hpp"

#include "abi.hpp"

namespace svm
{
	template<typename T, typename Abi>
	class simd_mask;
	template<typename T, typename Abi>
	class simd;

	namespace detail
	{
		template<std::size_t N, std::size_t I = 0, typename G>
		inline void generate(auto &data, G &&gen) noexcept
		{
			if constexpr (I != N)
			{
				data[I] = std::invoke(gen, std::integral_constant<std::size_t, I>());
				generate<N, I + 1>(data, std::forward<G>(gen));
			}
		}
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
	[[nodiscard]] inline simd_mask<T, Abi> operator&(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] && b[i];
		return result;
	}
	/** Preforms a bitwise OR on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator|(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] || b[i];
		return result;
	}
	/** Preforms a bitwise XOR on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator^(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b[i];
		return result;
	}

	/** Preforms a bitwise AND on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	inline simd_mask<T, Abi> &operator&=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a & b; }
	/** Preforms a bitwise OR on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	inline simd_mask<T, Abi> &operator|=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a | b; }
	/** Preforms a bitwise XOR on the elements of the masks \a a and \a b, and assigns the result to mask \a a. */
	template<typename T, typename Abi>
	inline simd_mask<T, Abi> &operator^=(simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept { return a = a ^ b; }

	/** Preforms a logical AND on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator&&(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] && b[i];
		return result;
	}
	/** Preforms a logical OR on the elements of the masks \a a and \a b, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator||(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] || b[i];
		return result;
	}

	/** Compares elements of masks \a a and \a b for equality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator==(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] == b[i];
		return result;
	}
	/** Compares elements of masks \a a and \a b for inequality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator!=(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result[i] = a[i] != b[i];
		return result;
	}

	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline bool all_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool result = true;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result &= mask[i];
		return result;
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline bool any_of(const simd_mask<T, Abi> &mask) noexcept
	{
		bool result = false;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result |= mask[i];
		return result;
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline bool none_of(const simd_mask<T, Abi> &mask) noexcept { return !any_of(mask); }
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline bool some_of(const simd_mask<T, Abi> &mask) noexcept
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
	[[nodiscard]] inline std::size_t popcount(const simd_mask<T, Abi> &mask) noexcept
	{
		std::size_t result = 0;
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i)
			result += mask[i];
		return result;
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<T, Abi> &mask) noexcept
	{
		for (std::size_t i = 0; i < simd_mask<T, Abi>::size(); ++i) if (mask[i]) return i;
		return simd_mask<T, Abi>::size();
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<T, Abi> &mask) noexcept
	{
		for (std::size_t i = simd_mask<T, Abi>::size(); i-- > 0;) if (mask[i]) return i;
		return simd_mask<T, Abi>::size();
	}

	SVM_EXT_NAMESPACE_OPEN
	/** Replaces elements of masks \a and \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
	template<typename T, typename Abi, typename M>
	[[nodiscard]] inline simd_mask<T, Abi> blend(const simd_mask<T, Abi> &a, const simd_mask<T, Abi> &b, const simd_mask<T, Abi> &m)
	{
		return blend(a, where(m, b));
	}
	SVM_EXT_NAMESPACE_CLOSE

	namespace detail
	{
		struct bool_wrapper
		{
			bool_wrapper() = delete;

			constexpr bool_wrapper(bool value) noexcept : value(value) {}
			constexpr operator bool() const noexcept { return value; }

			bool value;
		};
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

	SVM_EXT_NAMESPACE_OPEN
	/** Equivalent to `m ? b : a`. */
	template<typename T>
	[[nodiscard]] inline T blend(const T &a, const T &b, detail::bool_wrapper m) { return m ? b : a; }
	SVM_EXT_NAMESPACE_CLOSE

	template<detail::vectorizable T>
	class simd_mask<T, simd_abi::scalar>
	{
	public:
		using value_type = bool;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::scalar;
		using simd_type = simd<T, abi_type>;

		/** Returns width of the SIMD mask (always 1 for `simd_abi::scalar`). */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		static void assert_subscript([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size(), "simd_mask<T, simd_abi::scalar> subscript out of range");
		}

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the underlying boolean with \a value. */
		constexpr simd_mask(value_type value) noexcept : m_data(value) {}
		/** Initializes the underlying boolean from the value pointed to by \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying boolean from the value pointed to by \a mem. */
		template<typename Flags>
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { m_data = mem[0]; }
		/** Copies the underlying boolean to the value pointed to by \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { mem[0] = m_data; }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			assert_subscript(i);
			return reference{m_data};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			assert_subscript(i);
			return m_data;
		}

		[[nodiscard]] simd_mask operator!() const noexcept { return {!m_data}; }

	private:
		value_type m_data;
	};

	template<detail::vectorizable T, std::size_t N, std::size_t Align>
	class simd_mask<T, simd_abi::aligned_vector<N, Align>>
	{
	public:
		using value_type = bool;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::aligned_vector<N, Align>;
		using simd_type = simd<T, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		static void assert_subscript([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size(), "simd_mask<T, simd_abi::scalar> subscript out of range");
		}

		constexpr static std::size_t alignment = std::max(Align, alignof(bool[N]));

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the underlying elements with \a value. */
		constexpr simd_mask(value_type value) noexcept { std::fill_n(m_data, size(), value); }
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd_mask(const simd_mask<U, simd_abi::aligned_vector<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (alignment != alignof(value_type))
				other.copy_to(m_data, overaligned<alignment>);
			else
				other.copy_to(m_data, element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { std::copy_n(mem, size(), m_data); }
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { std::copy_n(m_data, size(), mem); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			assert_subscript(i);
			return reference{m_data[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			assert_subscript(i);
			return m_data[i];
		}

		[[nodiscard]] simd_mask operator!() const noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < size(); ++i)
				result[i] = !m_data[i];
			return result;
		}

	private:
		alignas(alignment) value_type m_data[size()];
	};

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
	[[nodiscard]] inline simd<T, Abi> operator+(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l + r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] + b[i];
		return result;
	}
	/** Subtracts elements of vector \a b from elements of vector \a a, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator-(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l - r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] - b[i];
		return result;
	}

	/** Adds elements of vector \a b to elements of vector \a a, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator+=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l += r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] += b[i];
		return a;
	}
	/** Subtracts elements of vector \a b from elements of vector \a a, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator-=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l -= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] -= b[i];
		return a;
	}

	/** Multiplies elements of vector \a a by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator*(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l * r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] * b[i];
		return result;
	}
	/** Divides elements of vector \a a by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator/(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l / r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] / b[i];
		return result;
	}
	/** Preforms a modulo operation of elements of vector \a a by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator%(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l % r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] % b[i];
		return result;
	}

	/** Multiplies elements of vector \a a by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator*=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l *= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] *= b[i];
		return a;
	}
	/** Divides elements of vector \a a by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator/=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l /= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] /= b[i];
		return a;
	}
	/** Preforms a modulo operation of elements of vector \a a by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator%=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l %= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] %= b[i];
		return a;
	}

	/** Preforms a bitwise AND between elements of vector \a a and elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator&(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l & r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] & b[i];
		return result;
	}
	/** Preforms a bitwise OR between elements of vector \a a and elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator|(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l | r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] | b[i];
		return result;
	}
	/** Preforms a bitwise XOR between elements of vector \a a and elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator^(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l ^ r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] ^ b[i];
		return result;
	}

	/** Preforms a bitwise AND between elements of vector \a a and elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator&=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l &= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] &= b[i];
		return a;
	}
	/** Preforms a bitwise OR between elements of vector \a a and elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator|=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l |= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] |= b[i];
		return a;
	}
	/** Preforms a bitwise XOR between elements of vector \a a and elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator^=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l ^= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] ^= b[i];
		return a;
	}

	/** Shifts elements of vector \a a left by the amount specified by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator<<(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l << r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] << b[i];
		return result;
	}
	/** Shifts elements of vector \a a right by the amount specified by elements of vector \a b, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator>>(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l >> r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >> b[i];
		return result;
	}

	/** Shifts elements of vector \a a left by the amount specified by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator<<=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l <<= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] <<= b[i];
		return a;
	}
	/** Shifts elements of vector \a a right by the amount specified by elements of vector \a b, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator>>=(simd<T, Abi> &a, const simd<T, Abi> &b) noexcept requires (requires(T l, T r){ l >>= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] >>= b[i];
		return a;
	}

	/** Shifts elements of vector \a a left by \a n, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator<<(const simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l << r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] << n;
		return result;
	}
	/** Shifts elements of vector \a a rught by \a n, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> operator>>(const simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l >> r; })
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >> n;
		return result;
	}

	/** Shifts elements of vector \a a left by \a n, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator<<=(simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l <<= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] <<= n;
		return a;
	}
	/** Shifts elements of vector \a a right by \a n, and returns reference to \a. */
	template<typename T, typename Abi>
	inline simd<T, Abi> &operator>>=(simd<T, Abi> &a, int n) noexcept requires (requires(T l, int r){ l >>= r; })
	{
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i) a[i] >>= n;
		return a;
	}

	/** Compares elements of vectors \a a and \a b for equality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator==(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] == b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for equality, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator!=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] != b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for less-than or equal, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator<=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] <= b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for greater-than or equal, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator>=(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] >= b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for less-than, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator<(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] < b[i];
		return result;
	}
	/** Compares elements of vectors \a a and \a b for greater-than, and returns the resulting mask. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd_mask<T, Abi> operator>(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd_mask<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = a[i] > b[i];
		return result;
	}

	namespace detail
	{
		template<std::size_t N, typename T, typename Abi, typename Op = std::plus<>>
		[[nodiscard]] inline T reduce_impl(const simd<T, Abi> &value, Op binary_op = {})
		{
			if constexpr (N == 1)
				return value[0];
			else
			{
				simd<T, simd_abi::deduce_t<T, N / 2, Abi>> a, b;

				/* Separate `value` into halves and reduce them separately. */
				for (std::size_t i = 0; i < N / 2; ++i)
					a[i] = value[0];
				for (std::size_t i = 0, j = N / 2; i < N / 2; ++i, ++j)
				{
					if (j < simd<T, Abi>::size())
						b[i] = value[j];
					else
						b[i] = T{};
				}
				return std::invoke(binary_op, reduce_impl<N / 2>(a, binary_op), reduce_impl<N / 2>(b, binary_op));
			}
		}
	}

	/** Calculates a reduction of all elements from \a value using \a binary_op. */
	template<typename T, typename Abi, typename Op = std::plus<>>
	[[nodiscard]] inline T reduce(const simd<T, Abi> &value, Op binary_op = {}) { return detail::reduce_impl<simd_size_v<T, Abi>>(value, binary_op); }

	SVM_EXT_NAMESPACE_OPEN
	/** Replaces elements of vectors \a and \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> blend(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd_mask<T, Abi> &m)
	{
		return blend(a, where(m, b));
	}
	SVM_EXT_NAMESPACE_CLOSE

	template<detail::vectorizable T>
	class simd<T, simd_abi::scalar>
	{
	public:
		using value_type = T;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::scalar;
		using mask_type = simd_mask<T, abi_type>;

		/** Returns width of the SIMD type (always 1 for `simd_abi::scalar`). */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		static void assert_subscript([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size(), "simd<T, simd_abi::scalar> subscript out of range");
		}

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
		void copy_to(U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { mem[0] = static_cast<U>(m_value); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			assert_subscript(i);
			return reference{m_value};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			assert_subscript(i);
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
		[[nodiscard]] mask_type operator~() const noexcept requires (requires { ~std::declval<value_type &>(); }) { return {~m_value}; }
		[[nodiscard]] simd operator+() const noexcept requires (requires { +std::declval<value_type &>(); }) { return {+m_value}; }
		[[nodiscard]] simd operator-() const noexcept requires (requires { -std::declval<value_type &>(); }) { return {-m_value}; }

	private:
		value_type m_value;
	};
}