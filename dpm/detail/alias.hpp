/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "utility.hpp"
#include <utility>

namespace dpm::detail
{
	/* SIMD mask elements are referenced via aliasing-enabled types. If a reference wrapper is used, RVO will bypass deleted
	 * constructors and may cause dangling references. As such, `reinterpret_cast` into a reference of a type that is allowed to alias
	 * others instead. Masks also need this due to the extension of boolean values into an integer mask. */
	template<typename T>
	class DPM_MAY_ALIAS basic_mask
	{
	public:
		basic_mask() = delete;
		basic_mask(const basic_mask &) = delete;

		constexpr DPM_FORCEINLINE operator bool() const noexcept { return static_cast<bool>(m_value); }

		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE basic_mask &operator=(U &&value) noexcept
		{
			m_value = extend_bool<T>(value);
			return *this;
		}

		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE basic_mask &operator|=(U &&value) noexcept
		{
			m_value |= extend_bool<T>(value);
			return *this;
		}
		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE basic_mask &operator&=(U &&value) noexcept
		{
			m_value &= extend_bool<T>(value);
			return *this;
		}
		template<std::convertible_to<bool> U>
		constexpr DPM_FORCEINLINE basic_mask &operator^=(U &&value) noexcept
		{
			m_value ^= extend_bool<T>(value);
			return *this;
		}

		friend constexpr void swap(basic_mask &a, basic_mask &b) noexcept { std::swap(a.m_value, b.m_value); }
		friend constexpr void swap(bool &a, basic_mask &b) noexcept { a = static_cast<bool>(std::exchange(b.m_value, extend_bool<T>(a))); }
		friend constexpr void swap(basic_mask &a, bool &b) noexcept { b = static_cast<bool>(std::exchange(a.m_value, extend_bool<T>(b))); }

	private:
		T m_value;
	};

	template<std::size_t>
	struct sized_mask;
	template<>
	struct sized_mask<1> { using type = basic_mask<std::int8_t>; };
	template<>
	struct sized_mask<2> { using type = basic_mask<std::int16_t>; };
	template<>
	struct sized_mask<4> { using type = basic_mask<std::int32_t>; };
	template<>
	struct sized_mask<8> { using type = basic_mask<std::int64_t>; };

	template<typename T>
	struct alias { typedef T DPM_MAY_ALIAS type; };
	template<typename T>
	using alias_t = typename alias<T>::type;
}