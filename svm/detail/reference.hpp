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
}