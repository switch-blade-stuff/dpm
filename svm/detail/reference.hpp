/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "define.hpp"
#include "utility.hpp"

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

		constexpr operator value_type() const & noexcept { return static_cast<value_type>(m_ref); }
		constexpr operator value_type() const && noexcept { return static_cast<value_type>(m_ref); }

		template<typename U>
		constexpr simd_reference operator=(U &&value) && noexcept requires (requires { std::declval<T &>() = static_cast<T>(std::forward<U>(value)); })
		{
			return simd_reference{m_ref = std::forward<U>(value)};
		}

		constexpr simd_reference operator++() && noexcept requires (requires { ++std::declval<T &>(); }) { return simd_reference{++m_ref}; }
		constexpr simd_reference operator--() && noexcept requires (requires { --std::declval<T &>(); }) { return simd_reference{--m_ref}; }

		constexpr value_type operator++(int) && noexcept requires (requires { ++std::declval<T &>()++; }) { return m_ref++; }
		constexpr value_type operator--(int) && noexcept requires (requires { ++std::declval<T &>()--; }) { return m_ref--; }

		template<typename U>
		constexpr simd_reference operator+=(U &&value) && noexcept requires (requires { std::declval<T &>() += std::forward<U>(value); })
		{
			return simd_reference{m_ref += std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator-=(U &&value) && noexcept requires (requires { std::declval<T &>() -= std::forward<U>(value); })
		{
			return simd_reference{m_ref -= std::forward<U>(value)};
		}

		template<typename U>
		constexpr simd_reference operator*=(U &&value) && noexcept requires (requires { std::declval<T &>() *= std::forward<U>(value); })
		{
			return simd_reference{m_ref *= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator/=(U &&value) && noexcept requires (requires { std::declval<T &>() /= std::forward<U>(value); })
		{
			return simd_reference{m_ref /= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator%=(U &&value) && noexcept requires (requires { std::declval<T &>() %= std::forward<U>(value); })
		{
			return simd_reference{m_ref %= std::forward<U>(value)};
		}

		template<typename U>
		constexpr simd_reference operator|=(U &&value) && noexcept requires (requires { std::declval<T &>() |= std::forward<U>(value); })
		{
			return simd_reference{m_ref |= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator&=(U &&value) && noexcept requires (requires { std::declval<T &>() &= std::forward<U>(value); })
		{
			return simd_reference{m_ref &= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator^=(U &&value) && noexcept requires (requires { std::declval<T &>() ^= std::forward<U>(value); })
		{
			return simd_reference{m_ref ^= std::forward<U>(value)};
		}

		template<typename U>
		constexpr simd_reference operator<<=(U &&value) && noexcept requires (requires { std::declval<T &>() <<= std::forward<U>(value); })
		{
			return simd_reference{m_ref <<= std::forward<U>(value)};
		}
		template<typename U>
		constexpr simd_reference operator>>=(U &&value) && noexcept requires (requires { std::declval<T &>() >>= std::forward<U>(value); })
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

	template<typename T>
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
		constexpr mask_reference operator=(U &&value) && noexcept { return mask_reference{m_ref = extend_bool<T>(value)}; }

		template<std::convertible_to<bool> U>
		constexpr mask_reference operator|=(U &&value) && noexcept { return mask_reference{m_ref |= extend_bool<T>(value)}; }
		template<std::convertible_to<bool> U>
		constexpr mask_reference operator&=(U &&value) && noexcept { return mask_reference{m_ref &= extend_bool<T>(value)}; }
		template<std::convertible_to<bool> U>
		constexpr mask_reference operator^=(U &&value) && noexcept { return mask_reference{m_ref ^= extend_bool<T>(value)}; }

		friend constexpr void swap(mask_reference &&a, mask_reference &&b) noexcept { std::swap(a.m_ref, b.m_ref); }
		friend constexpr void swap(value_type &a, mask_reference &&b) noexcept { a = static_cast<value_type>(std::exchange(b.m_ref, extend_bool<T>(a))); }
		friend constexpr void swap(mask_reference &&a, value_type &b) noexcept { b = static_cast<value_type>(std::exchange(a.m_ref, extend_bool<T>(b))); }

	private:
		T &m_ref;
	};
}