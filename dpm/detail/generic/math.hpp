/*
 * Created by switchblade on 2023-01-14.
 */

#pragma once

#include "type.hpp"

#ifndef DPM_USE_IMPORT

#include <cmath>

#ifdef DPM_HANDLE_ERRORS

#include <cerrno>
#include <cfenv>

#endif
#endif

#define DPM_MATHFUNC DPM_PURE DPM_VECTORCALL

namespace dpm
{
	namespace detail
	{
		template<typename... Ts>
		using promote_t = std::conditional_t<std::disjunction_v<std::is_same<Ts, long double>...>, long double, double>;
	}

#pragma region "basic operations"
	/** Calculates absolute value of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> abs(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::abs(x[i]);
		return result;
	}
	/** @copydoc abs */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> fabs(const simd<T, Abi> &x) noexcept { return abs(x); }

	/** Calculates floating-point remainder of elements in \a a divided by elements in vector \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmod(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fmod(a[i], b[i]);
		return result;
	}
	/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> remainder(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::remainder(a[i], b[i]);
		return result;
	}
	/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b, and stores the sign and
	 * at least 3 last bits of the division result in elements of \a quo. */
	template<typename T, typename Abi, typename QAbi>
	[[nodiscard]] inline simd<T, Abi> remquo(const simd<T, Abi> &a, const simd<T, Abi> &b, simd<int, QAbi> *quo) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::remquo(a[i], b[i], &((*quo)[i]));
		return result;
	}
	/** Calculates the maximum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmax(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fmax(a[i], b[i]);
		return result;
	}
	/** Calculates the minimum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmin(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fmin(a[i], b[i]);
		return result;
	}
	/** Returns the positive difference between elements of vectors \a a and \a b. Equivalent to `max(simd<T, Abi>{0}, a - b)`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fdim(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fdim(a[i], b[i]);
		return result;
	}
	/** Preforms linear interpolation or extrapolation between elements of vectors \a a and \a b using factor \a f */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> lerp(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &f) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::lerp(a[i], b[i], f[i]);
		return result;
	}

	/** @copydoc remainder
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted remainder(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{remainder(Promoted{a}, Promoted{b})}; }
	/** @copydoc remquo
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename QAbi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted remquo(const simd<T0, Abi> &a, const simd<T1, Abi> &b, simd<int, QAbi> *quo) noexcept { return Promoted{remquo(Promoted{a}, Promoted{b}, quo)}; }
	/** @copydoc fmax
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmax(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{fmax(Promoted{a}, Promoted{b})}; }
	/** @copydoc fmin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmin(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{fmin(Promoted{a}, Promoted{b})}; }
	/** @copydoc fdim
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fdim(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{fdim(Promoted{a}, Promoted{b})}; }
	/** @copydoc lerp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted lerp(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &f) noexcept { return Promoted{lerp(Promoted{a}, Promoted{b}, Promoted{f})}; }
	/** @copydoc fmod
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmod(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{fmod(Promoted{a}, Promoted{b})}; }

	/** Calculates floating-point remainder of elements in \a a divided by scalar \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmod(const simd<T, Abi> &a, T b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fmod(a[i], b);
		return result;
	}
	/** Calculates IEEE remainder of elements in \a a divided by scalar \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> remainder(const simd<T, Abi> &a, T b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::remainder(a[i], b);
		return result;
	}
	/** Calculates the maximum of elements in \a a and scalar \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmax(const simd<T, Abi> &a, T b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fmax(a[i], b);
		return result;
	}
	/** Calculates the minimum of elements in \a a and scalar \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fmin(const simd<T, Abi> &a, T b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fmin(a[i], b[i]);
		return result;
	}
	/** Returns the positive difference between elements of vector \a a and scalar \a b. Equivalent to `max(simd<T, Abi>{0}, a - b)`. */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> fdim(const simd<T, Abi> &a, T b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fdim(a[i], b);
		return result;
	}
	/** Preforms linear interpolation or extrapolation between elements of vectors \a a and \a b using scalar factor \a f */
	template<typename T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> lerp(const simd<T, Abi> &a, const simd<T, Abi> &b, T f) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::lerp(a[i], b[i], f);
		return result;
	}

	/** @copydoc fmod
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmod(const simd<T0, Abi> &a, T1 b) noexcept { return Promoted{fmod(Promoted{a}, detail::promote_t<T0, T1>{b})}; }
	/** @copydoc remainder
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted remainder(const simd<T0, Abi> &a, T1 b) noexcept { return Promoted{remainder(Promoted{a}, detail::promote_t<T0, T1>{b})}; }
	/** @copydoc fmax
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmax(const simd<T0, Abi> &a, T1 b) noexcept { return Promoted{fmax(Promoted{a}, detail::promote_t<T0, T1>{b})}; }
	/** @copydoc fmin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fmin(const simd<T0, Abi> &a, T1 b) noexcept { return Promoted{fmin(Promoted{a}, detail::promote_t<T0, T1>{b})}; }
	/** @copydoc fdim
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fdim(const simd<T0, Abi> &a, T1 b) noexcept { return Promoted{fdim(Promoted{a}, detail::promote_t<T0, T1>{b})}; }
	/** @copydoc lerp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted lerp(const simd<T0, Abi> &a, const simd<T1, Abi> &b, T2 f) noexcept { return Promoted{lerp(Promoted{a}, Promoted{b}, detail::promote_t<T0, T1, T2>{f})}; }

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b, and stores the sign and
		 * at least 3 last bits of the division result in elements of \a quo. Equivalent to the regular `remquo`, except
		 * that the quotent out parameter is taken by reference. */
		template<typename T, typename Abi>
		[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> remquo(const simd<T, Abi> &a, const simd<T, Abi> &b, rebind_simd_t<int, simd<T, Abi>> &quo) noexcept { return dpm::remquo(a, b, &quo); }
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fmadd(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return a * b + c; }
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fmsub(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return a * b - c; }
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fnmadd(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return c - a * b; }
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<typename T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> fnmsub(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return -(a * b) - c; }

		/** @copydoc remquo
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename Abi, typename QAbi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted remquo(const simd<T0, Abi> &a, const simd<T1, Abi> &b, simd<int, QAbi> &quo) noexcept { return Promoted{remquo(Promoted{a}, Promoted{b}, quo)}; }
		/** @copydoc fmadd
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted fmadd(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept { return Promoted{fmadd(Promoted{a}, Promoted{b}, Promoted{c})}; }
		/** @copydoc fmsub
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted fmsub(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept { return Promoted{fmsub(Promoted{a}, Promoted{b}, Promoted{c})}; }
		/** @copydoc fnmadd
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted fnmadd(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept { return Promoted{fnmadd(Promoted{a}, Promoted{b}, Promoted{c})}; }
		/** @copydoc fnmsub
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted fnmsub(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept { return Promoted{fnmsub(Promoted{a}, Promoted{b}, Promoted{c})}; }

		/** Converts the specified string to a corresponding NaN value as if via `simd<T, Abi>{std::nan[fl](str)}`. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> nan(const char *str) noexcept
		{
			if constexpr (std::same_as<T, double>)
				return simd < T, Abi > {std::nan(str)};
			else if constexpr (std::same_as<T, float>)
				return simd < T, Abi > {std::nanf(str)};
			else
				return simd < T, Abi > {std::nanl(str)};
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<typename T, typename Abi>
	[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> fma(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return ext::fmadd(a, b, c); }
	/** @copydoc fma
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename T2, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted fma(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept { return Promoted{fma(Promoted{a}, Promoted{b}, Promoted{c})}; }
#pragma endregion

#pragma region "power functions"
	/** Calculates square root of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> sqrt(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::sqrt(x[i]);
		return result;
	}
	/** Calculates cube root of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> cbrt(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::cbrt(x[i]);
		return result;
	}
	/** Calculates square root of the sum of elements in vectors \a a and \a b without causing over or underflow, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> hypot(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::hypot(a[i], b[i]);
		return result;
	}
	/** Raises elements of vector \a x to power specified by elements of vector \a p, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> pow(const simd<T, Abi> &x, const simd<T, Abi> &p) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::pow(x[i], p[i]);
		return result;
	}
	/** Raises elements of vector \a x to power \a p, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> pow(const simd<T, Abi> &x, T p) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::pow(x[i], p);
		return result;
	}

	/** @copydoc sqrt
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted sqrt(const simd<T, Abi> &x) noexcept { return Promoted{sqrt(Promoted{x})}; }
	/** @copydoc cbrt
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted cbrt(const simd<T, Abi> &x) noexcept { return Promoted{cbrt(Promoted{x})}; }
	/** @copydoc hypot
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted hypot(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{hypot(Promoted{a}, Promoted{b})}; }
	/** @copydoc pow
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted pow(const simd<T0, Abi> &x, const simd<T1, Abi> &p) noexcept { return Promoted{pow(Promoted{x}, Promoted{p})}; }
	/** @copydoc pow
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted pow(const simd<T0, Abi> &x, T1 p) noexcept { return Promoted{pow(Promoted{x}, detail::promote_t<T0, T1>{p})}; }

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates reciprocal of elements in vector \a x, and returns the resulting vector. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> rcp(const simd<T, Abi> &x) noexcept { return simd < T, Abi > {T{1}} / x; }
		/** @copydoc rcp
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted rcp(const simd<T, Abi> &x) noexcept { return Promoted{rcp(Promoted{x})}; }

		/** Calculates reciprocal square root of elements in vector \a x, and returns the resulting vector. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] DPM_FORCEINLINE simd<T, Abi> rsqrt(const simd<T, Abi> &x) noexcept { return rcp(sqrt(x)); }
		/** @copydoc rsqrt
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted rsqrt(const simd<T, Abi> &x) noexcept { return Promoted{rsqrt(Promoted{x})}; }
	}
#pragma endregion

#pragma region "trigonometric functions"
	namespace detail
	{
		[[nodiscard]] float DPM_PUBLIC DPM_MATHFUNC cot(float x) noexcept;
		[[nodiscard]] double DPM_PUBLIC DPM_MATHFUNC cot(double x) noexcept;
		[[nodiscard]] long double DPM_PUBLIC DPM_MATHFUNC cot(long double x) noexcept;
	}

	/** Calculates sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> sin(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::sin(x[i]);
		return result;
	}
	/** Calculates cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> cos(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::cos(x[i]);
		return result;
	}
	/** Calculates tangent of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> tan(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::tan(x[i]);
		return result;
	}
	/** Calculates arc-sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> asin(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::asin(x[i]);
		return result;
	}
	/** Calculates arc-cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> acos(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::acos(x[i]);
		return result;
	}
	/** Calculates arc-tangent of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> atan(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::atan(x[i]);
		return result;
	}
	/** Calculates arc-tangent of quotient of elements in vectors \a a and \a b, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> atan2(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::atan2(a[i], b[i]);
		return result;
	}

	/** @copydoc sin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted sin(const simd<T, Abi> &x) noexcept { return Promoted{sin(Promoted{x})}; }
	/** @copydoc cos
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted cos(const simd<T, Abi> &x) noexcept { return Promoted{cos(Promoted{x})}; }
	/** @copydoc tan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted tan(const simd<T, Abi> &x) noexcept { return Promoted{tan(Promoted{x})}; }
	/** @copydoc asin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted asin(const simd<T, Abi> &x) noexcept { return Promoted{asin(Promoted{x})}; }
	/** @copydoc acos
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted acos(const simd<T, Abi> &x) noexcept { return Promoted{acos(Promoted{x})}; }
	/** @copydoc atan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted atan(const simd<T, Abi> &x) noexcept { return Promoted{atan(Promoted{x})}; }
	/** @copydoc atan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted atan2(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return Promoted{atan2(Promoted{a}, Promoted{b})}; }

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates sine and cosine of elements in vector \a x, and assigns results to elements of \a out_sin and \a out_cos respectively. */
		template<typename T, typename Abi>
		DPM_FORCEINLINE void sincos(const simd<T, Abi> &x, simd<T, Abi> &out_sin, simd<T, Abi> &out_cos) noexcept
		{
			out_sin = sin(x);
			out_cos = cos(x);
		}

		/** Calculates cotangent of elements in vector \a x, and returns the resulting vector. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] inline simd<T, Abi> cot(const simd<T, Abi> &x) noexcept
		{
			simd < T, Abi > result = {};
			for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
				result[i] = detail::cot(x[i]);
			return result;
		}
		/** @copydoc cot
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
		[[nodiscard]] DPM_FORCEINLINE Promoted cot(const simd<T, Abi> &x) noexcept { return Promoted{atan(Promoted{x})}; }
	}
#pragma endregion

#pragma region "hyperbolic functions"
	/** Calculates hyperbolic sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> sinh(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::sinh(x[i]);
		return result;
	}
	/** Calculates hyperbolic cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> cosh(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::cosh(x[i]);
		return result;
	}
	/** Calculates hyperbolic tangent of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> tanh(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::tanh(x[i]);
		return result;
	}
	/** Calculates hyperbolic arc-sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> asinh(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::asinh(x[i]);
		return result;
	}
	/** Calculates hyperbolic arc-cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> acosh(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::acosh(x[i]);
		return result;
	}
	/** Calculates hyperbolic arc-tangent of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> atanh(const simd<T, Abi> &x) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::atanh(x[i]);
		return result;
	}

	/** @copydoc sinh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted sinh(const simd<T, Abi> &x) noexcept { return Promoted{sinh(Promoted{x})}; }
	/** @copydoc cosh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted cosh(const simd<T, Abi> &x) noexcept { return Promoted{cosh(Promoted{x})}; }
	/** @copydoc tanh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted tanh(const simd<T, Abi> &x) noexcept { return Promoted{tanh(Promoted{x})}; }
	/** @copydoc asinh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted asinh(const simd<T, Abi> &x) noexcept { return Promoted{asinh(Promoted{x})}; }
	/** @copydoc acosh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted acosh(const simd<T, Abi> &x) noexcept { return Promoted{acosh(Promoted{x})}; }
	/** @copydoc atanh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted atanh(const simd<T, Abi> &x) noexcept { return Promoted{atanh(Promoted{x})}; }
#pragma endregion

#pragma region "floating-point manipulation"
	/** Copies sign bit from elements of vector \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> copysign(const simd<T, Abi> &x, const simd<T, Abi> &sign) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::copysign(x[i], sign[i]);
		return result;
	}
	/** Copies sign bit from \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline simd<T, Abi> copysign(const simd<T, Abi> &x, T sign) noexcept
	{
		simd < T, Abi > result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::copysign(x[i], sign);
		return result;
	}

	/** @copydoc isnan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted copysign(const simd<T0, Abi> &x, const simd<T1, Abi> &sign) noexcept { return Promoted{copysign(Promoted{x}, Promoted{sign})}; }
	/** @copydoc isnan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE Promoted copysign(const simd<T0, Abi> &x, T1 sign) noexcept { return Promoted{copysign(Promoted{x}, detail::promote_t<T0, T1>{sign})}; }
#pragma endregion

#pragma region "classification"
	/** Classifies elements of vector \a x, returning one of one of `FP_INFINITE`, `FP_NAN`, `FP_NORMAL`, `FP_SUBNORMAL`, `FP_ZERO`. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline rebind_simd_t<int, simd<T, Abi>> fpclassify(const simd<T, Abi> &x) noexcept
	{
		rebind_simd_t<int, simd<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::fpclassify(x[i]);
		return result;
	}
	/** Determines is elements of \a x are finite and returns the resulting mask. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isfinite(const simd<T, Abi> &x) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isfinite(x[i]);
		return result;
	}
	/** Determines is elements of \a x are infinite and returns the resulting mask. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isinf(const simd<T, Abi> &x) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isinf(x[i]);
		return result;
	}
	/** Determines is elements of \a x are unordered NaN and returns the resulting mask. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isnan(const simd<T, Abi> &x) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isnan(x[i]);
		return result;
	}
	/** Determines is elements of \a x are normal and returns the resulting mask. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isnormal(const simd<T, Abi> &x) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isnormal(x[i]);
		return result;
	}
	/** Extracts a vector mask filled with sign bits of elements from vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type signbit(const simd<T, Abi> &x) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::signbit(x[i]);
		return result;
	}

	/** @copydoc fpclassify
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE rebind_simd_t<int, Promoted> fpclassify(const simd<T, Abi> &x) noexcept { return rebind_simd_t<int, Promoted>{fpclassify(Promoted{x})}; }
	/** @copydoc isfinite
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isfinite(const simd<T, Abi> &x) noexcept { return typename Promoted::mask_type{isfinite(Promoted{x})}; }
	/** @copydoc isinf
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isinf(const simd<T, Abi> &x) noexcept { return typename Promoted::mask_type{isinf(Promoted{x})}; }
	/** @copydoc isnan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isnan(const simd<T, Abi> &x) noexcept { return typename Promoted::mask_type{isnan(Promoted{x})}; }
	/** @copydoc isnormal
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isnormal(const simd<T, Abi> &x) noexcept { return typename Promoted::mask_type{isnormal(Promoted{x})}; }
	/** @copydoc signbit
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type signbit(const simd<T, Abi> &x) noexcept { return typename Promoted::mask_type{signbit(Promoted{x})}; }

	/** Determines is elements of \a a are greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isgreater(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isgreater(a[i], b[i]);
		return result;
	}
	/** Determines is elements of \a a are greater than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isgreaterequal(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isgreaterequal(a[i], b[i]);
		return result;
	}
	/** Determines is elements of \a a are less than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isless(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isless(a[i], b[i]);
		return result;
	}
	/** Determines is elements of \a a are less than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type islessequal(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::islessequal(a[i], b[i]);
		return result;
	}
	/** Determines is elements of \a a are less than or greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type islessgreater(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::islessgreater(a[i], b[i]);
		return result;
	}
	/** Determines is either elements of \a a or \a b are unordered and returns the resulting mask. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] inline typename simd<T, Abi>::mask_type isunordered(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		typename simd<T, Abi>::mask_type result = {};
		for (std::size_t i = 0; i < simd < T, Abi > ::size(); ++i)
			result[i] = std::isunordered(a[i], b[i]);
		return result;
	}

	/** @copydoc isgreater
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isgreater(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return typename Promoted::mask_type{isgreater(Promoted{a}, Promoted{b})}; }
	/** @copydoc isgreaterequal
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isgreaterequal(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return typename Promoted::mask_type{isgreaterequal(Promoted{a}, Promoted{b})}; }
	/** @copydoc isless
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isless(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return typename Promoted::mask_type{isless(Promoted{a}, Promoted{b})}; }
	/** @copydoc islessequal
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type islessequal(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return typename Promoted::mask_type{islessequal(Promoted{a}, Promoted{b})}; }
	/** @copydoc islessgreater
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type islessgreater(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return typename Promoted::mask_type{islessgreater(Promoted{a}, Promoted{b})}; }
	/** @copydoc isunordered
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename Promoted = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>>
	[[nodiscard]] DPM_FORCEINLINE typename Promoted::mask_type isunordered(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept { return typename Promoted::mask_type{isunordered(Promoted{a}, Promoted{b})}; }
#pragma endregion
}