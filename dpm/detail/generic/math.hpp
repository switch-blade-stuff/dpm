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
	namespace detail
	{
		template<typename... Ts>
		using promote_return = std::conditional_t<std::disjunction_v<std::is_same<Ts, long double>...>, long double, double>;
	}

#pragma region "basic operations"
	/** Calculates floating-point remainder of elements in \a a divided by elements in vector \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmod(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmod(a[i], b[i]);
		return result;
	}
	/** @copydoc fmod
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmod(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		return Promoted{fmod(Promoted{a}, Promoted{b})};
	}

	/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> remainder(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::remainder(a[i], b[i]);
		return result;
	}
	/** @copydoc remainder
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted remainder(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		return Promoted{remainder(Promoted{a}, Promoted{b})};
	}

	/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b, and stores the sign and
	 * at least 3 last bits of the division result in elements of \a quo. */
	template<typename T, typename Abi, typename QAbi>
	[[nodiscard]] inline simd<T, Abi> remquo(const simd<T, Abi> &a, const simd<T, Abi> &b, simd<int, QAbi> *quo) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::remquo(a[i], b[i], &((*quo)[i]));
		return result;
	}
	/** @copydoc remquo
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename QAbi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted remquo(const simd<T0, Abi> &a, const simd<T1, Abi> &b, simd<int, QAbi> *quo) noexcept
	{
		return Promoted{remquo(Promoted{a}, Promoted{b}, quo)};
	}

	/** Calculates the maximum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmax(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmax(a[i], b[i]);
		return result;
	}
	/** @copydoc fmax
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmax(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		return Promoted{fmax(Promoted{a}, Promoted{b})};
	}

	/** Calculates the minimum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmin(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmin(a[i], b[i]);
		return result;
	}
	/** @copydoc fmin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmin(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		return Promoted{fmin(Promoted{a}, Promoted{b})};
	}

	/** Returns the positive difference between x and y. Equivalent to `max(x - y, simd<T, Abi>{0})`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fdim(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fdim(a[i], b[i]);
		return result;
	}
	/** @copydoc fdim
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fdim(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		return Promoted{fdim(Promoted{a}, Promoted{b})};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b, and stores the sign and
		 * at least 3 last bits of the division result in elements of \a quo. Equivalent to the regular `remquo`, except
		 * that the quotent out parameter is taken by reference. */
		template<typename T, typename Abi>
		[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> remquo(const simd<T, Abi> &a, const simd<T, Abi> &b, rebind_simd_t<int, simd<T, Abi>> &quo) noexcept
		{
			return dpm::remquo(a, b, &quo);
		}
		/** @copydoc remquo
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename Abi, typename QAbi, typename Promoted = rebind_simd_t<detail::promote_return<T0, T1>, simd<T0, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted remquo(const simd<T0, Abi> &a, const simd<T1, Abi> &b, simd<int, QAbi> &quo) noexcept
		{
			return Promoted{remquo(Promoted{a}, Promoted{b}, quo)};
		}

		/** Converts the specified string to a corresponding NaN value as if via `simd<T, Abi>{std::nan[fl](str)}`. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> nan(const char *str) noexcept
		{
			if constexpr (std::same_as<T, double>)
				return simd<T, Abi>{std::nan(str)};
			else if constexpr (std::same_as<T, float>)
				return simd<T, Abi>{std::nanf(str)};
			else
				return simd<T, Abi>{std::nanl(str)};
		}
	}
#pragma endregion

	/** Calculates sine of elements in vector \a x, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> sin(const simd<T, Abi> &x) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::sin(x[i]);
		return result;
	}
	/** Calculates cosine of elements in vector \a x, and returns the resulting vector. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> cos(const simd<T, Abi> &x) noexcept
	{
		simd<T, Abi> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::cos(x[i]);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates sine and cosine of elements in vector \a x, and assigns results to elements of \a out_sin and \a out_cos respectively. */
		template<typename T, typename Abi>
		DPM_FORCEINLINE void sincos(const simd<T, Abi> &x, simd<T, Abi> &out_sin, simd<T, Abi> &out_cos) noexcept
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
	[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> fma(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept
	{
		return ext::fmadd(a, b, c);
	}
}