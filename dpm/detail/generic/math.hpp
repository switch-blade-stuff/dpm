/*
 * Created by switchblade on 2023-01-14.
 */

#pragma once

#include "type.hpp"

#ifndef DPM_USE_IMPORT

#include <cmath>

#endif

namespace dpm
{
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> sin(const simd<T, Abi> &x) noexcept
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::sin(x[i]);
		return result;
	}
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> cos(const simd<T, Abi> &x) noexcept
	{
		simd<T, Abi> result;
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::cos(x[i]);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<typename T, typename Abi>
		inline void sincos(const simd<T, Abi> &x, simd<T, Abi> &out_sin, simd<T, Abi> &out_cos) noexcept
		{
			out_sin = sin(x);
			out_cos = cos(x);
		}

		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fmadd(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept
		{
			return a * b + c;
		}
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fmsub(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept
		{
			return a * b - c;
		}
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fnmadd(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept
		{
			return c - a * b;
		}
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fnmsub(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept
		{
			return -(a * b) - c;
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fma(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept
	{
		return ext::fmadd(a, b, c);
	}
}