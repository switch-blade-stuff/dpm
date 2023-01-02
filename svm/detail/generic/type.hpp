/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../../flags.hpp"
#include "../reference.hpp"

#include "abi.hpp"

namespace svm
{
	/** @brief Type representing a data-parallel arithmetic type. */
	template<typename T, typename Abi>
	class simd
	{
	public:
		/* The standard mandates the default specialization to have all constructors & assignment operators be deleted.
		 * See N4808 - 9.6.1 [parallel.simd.overview] for details. */
		simd() = delete;
		simd(const simd &) = delete;
		simd &operator=(const simd &) = delete;
		~simd() = delete;
	};
	/** @brief Type representing a data-parallel mask type. */
	template<typename T, typename Abi>
	class simd_mask
	{
	public:
		/* The standard mandates the default specialization to have all constructors & assignment operators be deleted.
		 * See N4808 - 9.8.1 [parallel.simd.mask.overview] for details. */
		simd_mask() = delete;
		simd_mask(const simd_mask &) = delete;
		simd_mask &operator=(const simd_mask &) = delete;
		~simd_mask() = delete;
	};

	/** @brief Scalar overload of the data-parallel SIMD type. */
	template<typename T> requires std::is_arithmetic_v<T>
	class simd<T, simd_abi::scalar>
	{
	public:
		using value_type = T;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::scalar;
		using mask_type = simd_mask<T, abi_type>;

		/** Width of the SIMD type (always 1 for `simd_abi::scalar`). */
		static constexpr size_t size() noexcept { return abi_type::size; }

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
		constexpr simd(G &&gen) noexcept : simd(gen(std::integral_constant<std::size_t, 0>())) {}
		/** Initializes the underlying scalar as if via `static_cast<value_type>(mem[0])`. */
		template<typename U, typename Flags>
		constexpr simd(const U *mem, Flags) requires is_simd_flag_type_v<Flags> : m_value(static_cast<value_type>(mem[0])) {}

	private:
		value_type m_value;
	};

	template<typename T>
	struct is_simd<simd<T, simd_abi::scalar>> : std::true_type {};
}