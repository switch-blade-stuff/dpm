/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "define.hpp"

#ifndef SVM_USE_IMPORT

#include <utility>

#endif

namespace svm::detail
{
	template<typename T>
	class simd_reference
	{
	public:
		using value_type = T;

	public:
		simd_reference() = delete;
		simd_reference(const simd_reference &) = delete;

		/** @warning Internal use only! */
		constexpr explicit simd_reference(value_type &ref) noexcept : m_ref(ref) {}

		constexpr operator value_type() const noexcept { return static_cast<value_type>(m_ref); }

		template<typename U>
		constexpr simd_reference operator=(U &&value) && noexcept requires (requires { std::declval<value_type &>() = std::forward<U>(value); })
		{
			return simd_reference{m_ref = std::forward<U>(value)};
		}

		constexpr simd_reference operator++() && noexcept requires (requires { ++std::declval<value_type &>(); }) { return simd_reference{++m_ref}; }
		constexpr simd_reference operator--() && noexcept requires (requires { --std::declval<value_type &>(); }) { return simd_reference{--m_ref}; }

		constexpr value_type operator++(int) && noexcept requires (requires { ++std::declval<value_type &>()++; }) { return m_ref++; }
		constexpr value_type operator--(int) && noexcept requires (requires { ++std::declval<value_type &>()--; }) { return m_ref--; }

		template<typename U>
		constexpr simd_reference operator+=(U &&value) && noexcept requires (requires { std::declval<value_type &>() += std::forward<U>(value); })
		{
			return simd_reference{m_ref += std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator-=(U &&value) && noexcept requires (requires { std::declval<value_type &>() -= std::forward<U>(value); })
		{
			return simd_reference{m_ref -= std::forward<U>(value)};
		}

		template<typename U>
		constexpr simd_reference operator*=(U &&value) && noexcept requires (requires { std::declval<value_type &>() *= std::forward<U>(value); })
		{
			return simd_reference{m_ref *= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator/=(U &&value) && noexcept requires (requires { std::declval<value_type &>() /= std::forward<U>(value); })
		{
			return simd_reference{m_ref /= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator%=(U &&value) && noexcept requires (requires { std::declval<value_type &>() %= std::forward<U>(value); })
		{
			return simd_reference{m_ref %= std::forward<U>(value)};
		}

		template<typename U>
		constexpr simd_reference operator|=(U &&value) && noexcept requires (requires { std::declval<value_type &>() |= std::forward<U>(value); })
		{
			return simd_reference{m_ref |= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator&=(U &&value) && noexcept requires (requires { std::declval<value_type &>() &= std::forward<U>(value); })
		{
			return simd_reference{m_ref &= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator^=(U &&value) && noexcept requires (requires { std::declval<value_type &>() ^= std::forward<U>(value); })
		{
			return simd_reference{m_ref ^= std::forward<U>(value)};
		}

		template<typename U>
		constexpr simd_reference operator<<=(U &&value) && noexcept requires (requires { std::declval<value_type &>() <<= std::forward<U>(value); })
		{
			return simd_reference{m_ref <<= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator>>=(U &&value) && noexcept requires (requires { std::declval<value_type &>() >>= std::forward<U>(value); })
		{
			return simd_reference{m_ref >>= std::forward<U>(value)};
		}

		friend constexpr void swap(simd_reference &&a, simd_reference &&b) noexcept
		{
			using std::swap;
			swap(a.m_ref, b.m_ref);
		}
		friend constexpr void swap(value_type &a, simd_reference &&b) noexcept
		{
			using std::swap;
			swap(a, b.m_ref);
		}
		friend constexpr void swap(simd_reference &&a, value_type &b) noexcept
		{
			using std::swap;
			swap(a.m_ref, b);
		}

	private:
		value_type &m_ref;
	};

	template<typename T, T true_value>
	class mask_reference
	{
	public:
		using value_type = bool;

	public:
		mask_reference() = delete;
		mask_reference(const mask_reference &) = delete;

		/** @warning Internal use only! */
		constexpr explicit mask_reference(T &ref) noexcept : m_ref(ref) {}

		constexpr operator value_type() const noexcept { return static_cast<value_type>(m_ref); }

		template<std::convertible_to<bool> U>
		constexpr mask_reference operator=(U &&value) && noexcept
		{
			return mask_reference{m_ref = static_cast<bool>(value) ? true_value : T{}};
		}

		template<std::convertible_to<bool> U>
		constexpr mask_reference operator|=(U &&value) && noexcept
		{
			return mask_reference{m_ref |= static_cast<bool>(value) ? true_value : T{}};
		}
		template<std::convertible_to<bool> U>
		constexpr mask_reference operator&=(U &&value) && noexcept
		{
			return mask_reference{m_ref &= static_cast<bool>(value) ? true_value : T{}};
		}
		template<std::convertible_to<bool> U>
		constexpr mask_reference operator^=(U &&value) && noexcept
		{
			return mask_reference{m_ref ^= static_cast<bool>(value) ? true_value : T{}};
		}

		friend constexpr void swap(mask_reference &&a, mask_reference &&b) noexcept
		{
			using std::swap;
			swap(a.m_ref, b.m_ref);
		}
		friend constexpr void swap(value_type &a, mask_reference &&b) noexcept
		{
			const auto value = a ? true_value : T{};
			a = static_cast<value_type>(std::exchange(b.m_ref, value));
		}
		friend constexpr void swap(mask_reference &&a, value_type &b) noexcept
		{
			const auto value = b ? true_value : T{};
			b = static_cast<value_type>(std::exchange(a.m_ref, value));
		}

	private:
		T &m_ref;
	};
}