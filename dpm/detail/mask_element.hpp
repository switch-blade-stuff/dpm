/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "define.hpp"
#include "utility.hpp"

#ifndef DPM_USE_IMPORT

#include <utility>

#endif

namespace dpm::detail
{
	/* NOTE: Ugly hack to avoid potential dangling references. If a reference wrapper is used, RVO will bypass deleted constructors
	 * and may cause dangling references. As such, `reinterpret_cast` into a reference of a wrapper type instead. This does break
	 * the strict aliasing rule, but it is "fine", since the only member of `mask_element` is the aliased type, and we are dealing
	 * with x86 vector types that force us to break strict aliasing anyway. */
	template<typename T>
	class DPM_MAY_ALIAS mask_element
	{
	public:
		using value_type = bool;

	public:
		mask_element() = delete;
		mask_element(const mask_element &) = delete;

		constexpr operator value_type() const noexcept { return static_cast<value_type>(m_value); }

		template<std::convertible_to<bool> U>
		constexpr mask_element &operator=(U &&value) & noexcept
		{
			m_value = extend_bool<T>(value);
			return *this;
		}

		template<std::convertible_to<bool> U>
		constexpr mask_element &operator|=(U &&value) & noexcept
		{
			m_value |= extend_bool<T>(value);
			return *this;
		}
		template<std::convertible_to<bool> U>
		constexpr mask_element &operator&=(U &&value) & noexcept
		{
			m_value &= extend_bool<T>(value);
			return *this;
		}
		template<std::convertible_to<bool> U>
		constexpr mask_element &operator^=(U &&value) & noexcept
		{
			m_value ^= extend_bool<T>(value);
			return *this;
		}

		friend constexpr void swap(mask_element &a, mask_element &b) noexcept { std::swap(a.m_value, b.m_value); }
		friend constexpr void swap(value_type &a, mask_element &b) noexcept { a = static_cast<value_type>(std::exchange(b.m_value, extend_bool<T>(a))); }
		friend constexpr void swap(mask_element &a, value_type &b) noexcept { b = static_cast<value_type>(std::exchange(a.m_value, extend_bool<T>(b))); }

	private:
		T m_value;
	};
}