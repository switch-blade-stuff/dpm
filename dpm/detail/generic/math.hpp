/*
 * Created by switchblade on 2023-01-14.
 */

#pragma once

#include "type.hpp"

#include <cmath>

#ifdef DPM_HANDLE_ERRORS

#include <cfenv>

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
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> abs(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::abs(x[i]);
		return result;
	}
	/** @copydoc abs */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> fabs(const simd<T, Abi> &x) noexcept { return abs(x); }

	/** Calculates floating-point remainder of elements in \a a divided by elements in vector \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fmod(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmod(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> remainder(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::remainder(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b, and stores the sign and
	 * at least 3 last bits of the division result in elements of \a out_quo. */
	template<typename T, typename Abi, typename QAbi>
	[[nodiscard]] constexpr simd<T, Abi> remquo(const simd<T, Abi> &a, const simd<T, Abi> &b, simd<int, QAbi> *out_quo) noexcept
	{
		alignas(simd<int, Abi>) std::array<int, simd_size_v<int, Abi>> quo = {};
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::remquo(a[i], b[i], quo.data() + i);

		out_quo->copy_from(quo.data(), vector_aligned);
		return {result.data(), vector_aligned};
	}
	/** Calculates the maximum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fmax(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmax(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates the minimum of elements in \a a and \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fmin(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmin(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Returns the positive difference between elements of vectors \a a and \a b. Equivalent to `max(simd<T, Abi>{0}, a - b)`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fdim(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fdim(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Preforms linear interpolation or extrapolation between elements of vectors \a a and \a b using factor \a f */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> lerp(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &f) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::lerp(a[i], b[i], f[i]);
		return {result.data(), vector_aligned};
	}

	/** @copydoc remainder
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto remainder(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return remainder(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc remquo
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi, typename QAbi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto remquo(const simd<T0, Abi> &a, const simd<T1, Abi> &b, simd<int, QAbi> *quo) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return remquo(promoted_t{a}, promoted_t{b}, quo);
	}
	/** @copydoc fmax
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fmax(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fmax(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc fmin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fmin(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fmin(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc fdim
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fdim(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fdim(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc lerp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename T2, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto lerp(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &f) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>;
		return lerp(promoted_t{a}, promoted_t{b}, promoted_t{f});
	}
	/** @copydoc fmod
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fmod(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fmod(promoted_t{a}, promoted_t{b});
	}

	/** Calculates floating-point remainder of elements in \a a divided by scalar \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fmod(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmod(a[i], b);
		return {result.data(), vector_aligned};
	}
	/** Calculates IEEE remainder of elements in \a a divided by scalar \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> remainder(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::remainder(a[i], b);
		return {result.data(), vector_aligned};
	}
	/** Calculates the maximum of elements in \a a and scalar \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fmax(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmax(a[i], b);
		return {result.data(), vector_aligned};
	}
	/** Calculates the minimum of elements in \a a and scalar \a b, respecting the NaN propagation
	 * as specified in IEC 60559 (ordered values are always selected over unordered). */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fmin(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fmin(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Returns the positive difference between elements of vector \a a and scalar \a b. Equivalent to `max(simd<T, Abi>{0}, a - b)`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> fdim(const simd<T, Abi> &a, T b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fdim(a[i], b);
		return {result.data(), vector_aligned};
	}
	/** Preforms linear interpolation or extrapolation between elements of vectors \a a and \a b using scalar factor \a f */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> lerp(const simd<T, Abi> &a, const simd<T, Abi> &b, T f) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::lerp(a[i], b[i], f);
		return {result.data(), vector_aligned};
	}

	/** @copydoc fmod
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fmod(const simd<T0, Abi> &a, T1 b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fmod(promoted_t{a}, detail::promote_t<T0, T1>{b});
	}
	/** @copydoc remainder
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto remainder(const simd<T0, Abi> &a, T1 b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return remainder(promoted_t{a}, detail::promote_t<T0, T1>{b});
	}
	/** @copydoc fmax
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fmax(const simd<T0, Abi> &a, T1 b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fmax(promoted_t{a}, detail::promote_t<T0, T1>{b});
	}
	/** @copydoc fmin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fmin(const simd<T0, Abi> &a, T1 b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fmin(promoted_t{a}, detail::promote_t<T0, T1>{b});
	}
	/** @copydoc fdim
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fdim(const simd<T0, Abi> &a, T1 b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return fdim(promoted_t{a}, detail::promote_t<T0, T1>{b});
	}
	/** @copydoc lerp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename T2, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto lerp(const simd<T0, Abi> &a, const simd<T1, Abi> &b, T2 f) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>;
		return lerp(promoted_t{a}, promoted_t{b}, detail::promote_t<T0, T1, T2>{f});
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates IEEE remainder of elements in \a a divided by elements in vector \a b, and stores the sign and
		 * at least 3 last bits of the division result in elements of \a quo. Equivalent to the regular `remquo`, except
		 * that the quotent out parameter is taken by reference. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> remquo(const simd<T, Abi> &a, const simd<T, Abi> &b, rebind_simd_t<int, simd<T, Abi>> &quo) noexcept { return dpm::remquo(a, b, &quo); }
		/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> fmadd(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return a * b + c; }
		/** Returns a result of fused multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `a * b - c`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> fmsub(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return a * b - c; }
		/** Returns a result of fused negate-multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) + c`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> fnmadd(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return c - a * b; }
		/** Returns a result of fused negate-multiply-sub operation on elements of \a a, \a b, and \a c. Equivalent to `-(a * b) - c`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr simd<T, Abi> fnmsub(const simd<T, Abi> &a, const simd<T, Abi> &b, const simd<T, Abi> &c) noexcept { return -(a * b) - c; }

		/** @copydoc remquo
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename Abi, typename QAbi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto remquo(const simd<T0, Abi> &a, const simd<T1, Abi> &b, simd<int, QAbi> &out_quo) noexcept
		{
			using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
			return remquo(promoted_t{a}, promoted_t{b}, out_quo);
		}
		/** @copydoc fmadd
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto fmadd(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept
		{
			using promoted_t = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>;
			return fmadd(promoted_t{a}, promoted_t{b}, promoted_t{c});
		}
		/** @copydoc fmsub
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto fmsub(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept
		{
			using promoted_t = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>;
			return fmsub(promoted_t{a}, promoted_t{b}, promoted_t{c});
		}
		/** @copydoc fnmadd
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto fnmadd(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept
		{
			using promoted_t = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>;
			return fnmadd(promoted_t{a}, promoted_t{b}, promoted_t{c});
		}
		/** @copydoc fnmsub
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T0, typename T1, typename T2, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto fnmsub(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept
		{
			using promoted_t = rebind_simd_t<detail::promote_t<T0, T1, T2>, simd<T0, Abi>>;
			return fnmsub(promoted_t{a}, promoted_t{b}, promoted_t{c});
		}

		/** Converts the specified string to a corresponding NaN value as if via `simd<T, Abi>{std::nan[fl](str)}`. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> nan(const char *str) noexcept
		{
			if constexpr (std::same_as<T, double>)
				return simd<T, Abi>{std::nan(str)};
			else if constexpr (std::same_as<T, float>)
				return simd<T, Abi>{std::nanf(str)};
			else
				return simd<T, Abi>{std::nanl(str)};
		}
	}

	/** Returns a result of fused multiply-add operation on elements of \a a, \a b, and \a c. Equivalent to `a * b + c`. */
	template<typename T0, typename T1, typename T2, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fma(const simd<T0, Abi> &a, const simd<T1, Abi> &b, const simd<T2, Abi> &c) noexcept { return ext::fmadd(a, b, c); }
#pragma endregion

#pragma region "power functions"
	/** Calculates square root of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> sqrt(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::sqrt(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates cube root of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> cbrt(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::cbrt(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates square root of the sum of elements in vectors \a a and \a b without causing over or underflow. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> hypot(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::hypot(a[i], b[i]);
		return {result.data(), vector_aligned};
	}
	/** Raises elements of vector \a x to power specified by elements of vector \a p. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> pow(const simd<T, Abi> &x, const simd<T, Abi> &p) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::pow(x[i], p[i]);
		return {result.data(), vector_aligned};
	}
	/** Raises elements of vector \a x to power \a p. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> pow(const simd<T, Abi> &x, T p) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::pow(x[i], p);
		return {result.data(), vector_aligned};
	}

	/** @copydoc sqrt
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto sqrt(const simd<T, Abi> &x) noexcept { return sqrt(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc cbrt
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto cbrt(const simd<T, Abi> &x) noexcept { return cbrt(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc hypot
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto hypot(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return hypot(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc pow
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto pow(const simd<T0, Abi> &x, const simd<T1, Abi> &p) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return pow(promoted_t{x}, promoted_t{p});
	}
	/** @copydoc pow
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto pow(const simd<T0, Abi> &x, T1 p) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return pow(promoted_t{x}, detail::promote_t<T0, T1>{p});
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates reciprocal of elements in vector \a x. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> rcp(const simd<T, Abi> &x) noexcept { return simd<T, Abi>{T{1}} / x; }
		/** @copydoc rcp
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto rcp(const simd<T, Abi> &x) noexcept { return rcp(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }

		/** Calculates reciprocal square root of elements in vector \a x. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> rsqrt(const simd<T, Abi> &x) noexcept { return rcp(sqrt(x)); }
		/** @copydoc rsqrt
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto rsqrt(const simd<T, Abi> &x) noexcept { return rsqrt(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	}
#pragma endregion

#pragma region "exponential functions"
	/** Raises *e* (Euler's number) to the power specified by elements of \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> exp(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::exp(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Raises `2` to the power specified by elements of \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> exp2(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::exp2(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Raises *e* (Euler's number) to the power specified by elements of \a x, and subtracts `1`. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> expm1(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::expm1(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates natural (base *e*) logarithm of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> log(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::log(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates common (base 10) logarithm of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> log10(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::log10(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates binary (base 2) logarithm of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> log2(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::log2(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates natural (base *e*) logarithm of elements in vector \a x plus `1`. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> log1p(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::log1p(x[i]);
		return {result.data(), vector_aligned};
	}

	/** @copydoc exp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto exp(const simd<T, Abi> &x) noexcept { return exp(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc exp2
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto exp2(const simd<T, Abi> &x) noexcept { return exp2(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc expm1
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto expm1(const simd<T, Abi> &x) noexcept { return expm1(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc log
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto log(const simd<T, Abi> &x) noexcept { return log(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc log10
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto log10(const simd<T, Abi> &x) noexcept { return log10(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc log2
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto log2(const simd<T, Abi> &x) noexcept { return log2(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc log1p
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto log1p(const simd<T, Abi> &x) noexcept { return log1p(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
#pragma endregion

#pragma region "trigonometric functions"
	/** Calculates sine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> sin(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::sin(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates cosine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> cos(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::cos(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates tangent of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> tan(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::tan(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates arc-sine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> asin(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::asin(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates arc-cosine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> acos(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::acos(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates arc-tangent of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> atan(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::atan(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates arc-tangent of quotient of elements in vectors \a a and \a b. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> atan2(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::atan2(a[i], b[i]);
		return {result.data(), vector_aligned};
	}

	/** @copydoc sin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto sin(const simd<T, Abi> &x) noexcept { return sin(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc cos
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto cos(const simd<T, Abi> &x) noexcept { return cos(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc tan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto tan(const simd<T, Abi> &x) noexcept { return tan(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc asin
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto asin(const simd<T, Abi> &x) noexcept { return asin(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc acos
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto acos(const simd<T, Abi> &x) noexcept { return acos(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc atan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto atan(const simd<T, Abi> &x) noexcept { return atan(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc atan
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto atan2(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return atan2(promoted_t{a}, promoted_t{b});
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates sine and cosine of elements in vector \a x, and assigns results to elements of \a out_sin and \a out_cos respectively. */
		template<std::floating_point T, typename Abi>
		constexpr DPM_FORCEINLINE void sincos(const simd<T, Abi> &x, simd<T, Abi> &out_sin, simd<T, Abi> &out_cos) noexcept
		{
#if defined(__GNUC__) && defined(__has_builtin)
#if (__has_builtin(__builtin_sincosf) && __has_builtin(__builtin_sincos) && __has_builtin(__builtin_sincosl)) || (defined(_GNU_SOURCE) && 0) /* GNU sincos function is not optimized well by compilers. */
#define DPM_USE_SINCOS
#endif
#endif
#ifdef DPM_USE_SINCOS
			if (!std::is_constant_evaluated())
				for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
				{
					T sin, cos;
					if constexpr (std::same_as<T, float>)
					{
#if __has_builtin(__builtin_sincosf)
						__builtin_sincosf(x, &sin, &cos);
#else
						::sincosf(x[i], &sin, &cos);
#endif
					}
					else if constexpr (std::same_as<T, long double>)
					{
#if __has_builtin(__builtin_sincosl)
						__builtin_sincosl(x, &sin, &cos);
#else
						::sincosl(x[i], &sin, &cos);
#endif
					}
					else
					{
#if __has_builtin(__builtin_sincos)
						__builtin_sincos(x, &sin, &cos);
#else
						::sincos(x[i], &sin, &cos);
#endif
					}
					out_sin[i] = sin;
					out_cos[i] = cos;
				}
			else
#else
			{
				out_sin = sin(x);
				out_cos = cos(x);
			}
#endif
		}
	}
#pragma endregion

#pragma region "hyperbolic functions"
	/** Calculates hyperbolic sine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> sinh(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::sinh(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates hyperbolic cosine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> cosh(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::cosh(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates hyperbolic tangent of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> tanh(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::tanh(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates hyperbolic arc-sine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> asinh(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::asinh(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates hyperbolic arc-cosine of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> acosh(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::acosh(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates hyperbolic arc-tangent of elements in vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> atanh(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::atanh(x[i]);
		return {result.data(), vector_aligned};
	}

	/** @copydoc sinh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto sinh(const simd<T, Abi> &x) noexcept { return sinh(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc cosh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto cosh(const simd<T, Abi> &x) noexcept { return cosh(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc tanh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto tanh(const simd<T, Abi> &x) noexcept { return tanh(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc asinh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto asinh(const simd<T, Abi> &x) noexcept { return asinh(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc acosh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto acosh(const simd<T, Abi> &x) noexcept { return acosh(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc atanh
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto atanh(const simd<T, Abi> &x) noexcept { return atanh(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
#pragma endregion

#pragma region "error functions"
	/** Calculates the error function of elements in \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> erf(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::erf(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates the complementary error function of elements in \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> erfc(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::erfc(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates the gamma function of elements in \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> tgamma(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::tgamma(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Calculates the natural logarithm of the absolute value of the gamma function of elements in \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> lgamma(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::lgamma(x[i]);
		return {result.data(), vector_aligned};
	}

	/** @copydoc erf
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto erf(const simd<T, Abi> &x) noexcept { return erf(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc erfc
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto erfc(const simd<T, Abi> &x) noexcept { return erfc(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc tgamma
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto tgamma(const simd<T, Abi> &x) noexcept { return tgamma(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc lgamma
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto lgamma(const simd<T, Abi> &x) noexcept { return lgamma(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
#pragma endregion

#pragma region "nearest integer functions"
	/** Rounds elements of vector \a x to nearest integer not less than the element's value. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> ceil(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::ceil(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Rounds elements of vector \a x to nearest integer not greater than the element's value. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> floor(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::floor(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Rounds elements of vector \a x to integer with truncation. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> trunc(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::trunc(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Rounds elements of vector \a x to integer using current rounding mode. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> nearbyint(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::nearbyint(x[i]);
		return {result.data(), vector_aligned};
	}

	/** Rounds elements of vector \a x to nearest integer rounding away from zero. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> round(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::round(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Casts elements of vector \a x to `long` rounding away from zero. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr rebind_simd_t<long, simd<T, Abi>> lround(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<long, Abi>) std::array<long, simd_size_v<long, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::lround(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Casts elements of vector \a x to `long long` rounding away from zero. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr rebind_simd_t<long long, simd<T, Abi>> llround(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<long long, Abi>) std::array<long long, simd_size_v<long long, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::llround(x[i]);
		return {result.data(), vector_aligned};
	}

	/** Rounds elements of vector \a x to nearest integer using current rounding mode with exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> rint(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<long, Abi>) std::array<long, simd_size_v<long, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::rint(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Casts elements of vector \a x to `long` using current rounding mode with exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr rebind_simd_t<long, simd<T, Abi>> lrint(const simd<T, Abi> &x) noexcept
	{
		rebind_simd_t<long, simd<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::lrint(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Casts elements of vector \a x to `long long` using current rounding mode with exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr rebind_simd_t<long long, simd<T, Abi>> llrint(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<long long, Abi>) std::array<long long, simd_size_v<long long, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::llrint(x[i]);
		return {result.data(), vector_aligned};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Casts elements of vector \a x to integer of type \a I rounding away from zero. */
		template<std::signed_integral I, std::floating_point T, typename Abi>
		[[nodiscard]] constexpr rebind_simd_t<I, simd<T, Abi>> iround(const simd<T, Abi> &x) noexcept
		{
			rebind_simd_t<I, simd<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			{
				if constexpr (sizeof(I) <= sizeof(long))
					result[i] = static_cast<I>(std::lround(x[i]));
				else if constexpr (!std::same_as<I, long long>)
					result[i] = static_cast<I>(std::round(x[i]));
				else
					result[i] = std::llround(x[i]);
			}
			return result;
		}

		/** Casts elements of vector \a x to integer of type \a I using current rounding mode with exceptions. */
		template<std::signed_integral I, std::floating_point T, typename Abi>
		[[nodiscard]] constexpr rebind_simd_t<I, simd<T, Abi>> irint(const simd<T, Abi> &x) noexcept
		{
			rebind_simd_t<I, simd<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			{
				if constexpr (sizeof(I) <= sizeof(long))
					result[i] = static_cast<I>(std::lrint(x[i]));
				else if constexpr (!std::same_as<I, long long>)
					result[i] = static_cast<I>(std::rint(x[i]));
				else
					result[i] = std::llrint(x[i]);
			}
			return result;
		}

		/** Casts elements of vector \a x to signed integer type \a I with truncation. */
		template<std::signed_integral I, std::floating_point T, typename Abi>
		[[nodiscard]] constexpr rebind_simd_t<I, simd<T, Abi>> itrunc(const simd<T, Abi> &x) noexcept
		{
			rebind_simd_t<I, simd<T, Abi>> result = {};
			for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
				result[i] = static_cast<I>(std::trunc(x[i]));
			return result;
		}
		/** Casts elements of vector \a x to `long` with truncation. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE rebind_simd_t<long, simd<T, Abi>> ltrunc(const simd<T, Abi> &x) noexcept { return itrunc<long>(x); }
		/** Casts elements of vector \a x to `long long` with truncation. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE rebind_simd_t<long long, simd<T, Abi>> lltrunc(const simd<T, Abi> &x) noexcept { return itrunc<long long>(x); }
	}
#pragma endregion

#pragma region "floating-point manipulation"
	/** Decomposes elements of vector \a x into a normalized fraction and a power-of-two exponent, stores the exponent in \a out_exp, and returns the fraction. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> frexp(const simd<T, Abi> &x, simd<int, Abi> *out_exp) noexcept
	{
		alignas(simd<int, Abi>) std::array<int, simd_size_v<int, Abi>> exp = {};
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::frexp(x[i], exp.data() + i);

		out_exp->copy_from(exp.data(), vector_aligned);
		return {result.data(), vector_aligned};
	}
	/** Decomposes elements of vector \a x into integral and fractional parts, returning the fractional and storing the integral in \a out_i. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> modf(const simd<T, Abi> &x, simd<T, Abi> *out_i) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> iprt = {};

		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::modf(x[i], iprt.data() + i);

		out_i->copy_from(iprt.data(), vector_aligned);
		return {result.data(), vector_aligned};
	}
	/** Multiplies elements of vector \a x by `2` raised to power specified by elements of vector \a exp. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> ldexp(const simd<T, Abi> &x, const simd<int, Abi> &exp) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::ldexp(x[i], exp[i]);
		return {result.data(), vector_aligned};
	}
	/** Multiplies elements of vector \a x by `FLT_RADIX` raised to power specified by elements of vector \a exp. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> scalbn(const simd<T, Abi> &x, const simd<int, Abi> &exp) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::scalbn(x[i], exp[i]);
		return {result.data(), vector_aligned};
	}
	/** @copydoc scalbn */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> scalbln(const simd<T, Abi> &x, const simd<long, Abi> &exp) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::scalbln(x[i], exp[i]);
		return {result.data(), vector_aligned};
	}
	/** Extracts unbiased exponent of elements in vector \a x as integers. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr rebind_simd_t<int, simd<T, Abi>> ilogb(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<int, Abi>) std::array<int, simd_size_v<int, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::ilogb(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Extracts unbiased exponent of elements in vector \a x as floats. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> logb(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::logb(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Finds next representable value from elements of vector \a from to elements of vector \a to. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> nextafter(const simd<T, Abi> &from, const simd<T, Abi> &to) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::nextafter(from[i], to[i]);
		return {result.data(), vector_aligned};
	}
	/** Finds next representable value from elements of vector \a from to elements of vector \a to without loss of precision. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> nexttoward(const simd<T, Abi> &from, const simd<long double, Abi> &to) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::nexttoward(from[i], to[i]);
		return {result.data(), vector_aligned};
	}
	/** Copies sign bit from elements of vector \a sign to elements of vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> copysign(const simd<T, Abi> &x, const simd<T, Abi> &sign) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::copysign(x[i], sign[i]);
		return {result.data(), vector_aligned};
	}

	/** @copydoc frexp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto frexp(const simd<T, Abi> &x, simd<int, Abi> *exp) noexcept { return frexp(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc ldexp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto ldexp(const simd<T, Abi> &x, const simd<int, Abi> &exp) noexcept { return ldexp(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc scalbn
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto scalbn(const simd<T, Abi> &x, const simd<int, Abi> &exp) noexcept { return scalbn(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc scalbln
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto scalbln(const simd<T, Abi> &x, const simd<long, Abi> &exp) noexcept { return scalbln(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc ilogb
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto ilogb(const simd<T, Abi> &x) noexcept { return ilogb(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc logb
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto logb(const simd<T, Abi> &x) noexcept { return logb(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc nextafter
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto nextafter(const simd<T0, Abi> &from, const simd<T1, Abi> &to) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return nextafter(promoted_t{from}, promoted_t{to});
	}
	/** @copydoc nextafter
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto nexttoward(const simd<T, Abi> &from, const simd<long double, Abi> &to) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>;
		return nexttoward(promoted_t{from}, to);
	}
	/** @copydoc copysign
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto copysign(const simd<T0, Abi> &x, const simd<T1, Abi> &sign) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return copysign(promoted_t{x}, promoted_t{sign});
	}

	/** Multiplies elements of vector \a x by `2` raised to power \a exp. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> ldexp(const simd<T, Abi> &x, int exp) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::ldexp(x[i], exp);
		return {result.data(), vector_aligned};
	}
	/** Multiplies elements of vector \a x by `FLT_RADIX` raised to power \a exp. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> scalbn(const simd<T, Abi> &x, int exp) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::scalbn(x[i], exp);
		return {result.data(), vector_aligned};
	}
	/** @copydoc scalbn */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> scalbln(const simd<T, Abi> &x, long exp) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::scalbln(x[i], exp);
		return {result.data(), vector_aligned};
	}
	/** Copies sign bit from \a sign to elements of vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr simd<T, Abi> copysign(const simd<T, Abi> &x, T sign) noexcept
	{
		alignas(simd<T, Abi>) std::array<T, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::copysign(x[i], sign);
		return {result.data(), vector_aligned};
	}

	/** @copydoc ldexp
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto ldexp(const simd<T, Abi> &x, int exp) noexcept { return ldexp(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc scalbn
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto scalbn(const simd<T, Abi> &x, int exp) noexcept { return scalbn(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc scalbln
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto scalbln(const simd<T, Abi> &x, long exp) noexcept { return scalbln(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}, exp); }
	/** @copydoc copysign
	 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto copysign(const simd<T0, Abi> &x, T1 sign) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return copysign(promoted_t{x}, detail::promote_t<T0, T1>{sign});
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Decomposes elements of vector \a x into a normalized fraction and a power-of-two exponent, stores the exponent in \a exp, and returns the fraction. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> frexp(const simd<T, Abi> &x, simd<int, Abi> &exp) noexcept { return frexp(x, &exp); }
		/** Decomposes elements of vector \a x into integral and fractional parts, returning the fractional and storing the integral in \a ip. */
		template<std::floating_point T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE simd<T, Abi> modf(const simd<T, Abi> &x, simd<T, Abi> &ip) noexcept { return modf(x, &ip); }

		/** @copydoc frexp
		 * @note Arguments and return type are promoted to `double`, or `long double` if one of the arguments is `long double`. */
		template<typename T, typename Abi>
		[[nodiscard]] constexpr DPM_FORCEINLINE auto frexp(const simd<T, Abi> &x, simd<int, Abi> &exp) noexcept
		{
			using promoted_t = rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>;
			return frexp(promoted_t{x}, exp);
		}
	}
#pragma endregion

#pragma region "classification"
	/** Classifies elements of vector \a x, returning one of one of `FP_INFINITE`, `FP_NAN`, `FP_NORMAL`, `FP_SUBNORMAL`, `FP_ZERO`. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr rebind_simd_t<int, simd<T, Abi>> fpclassify(const simd<T, Abi> &x) noexcept
	{
		alignas(simd<int, Abi>) std::array<int, simd_size_v<int, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::fpclassify(x[i]);
		return {result.data(), vector_aligned};
	}
	/** Determines if elements of \a x are finite. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isfinite(const simd<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isfinite(x[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a x are infinite. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isinf(const simd<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isinf(x[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a x are unordered NaN. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isnan(const simd<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isnan(x[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a x are normal. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isnormal(const simd<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isnormal(x[i]);
		return {result.data(), element_aligned};
	}
	/** Extracts a vector mask filled with sign bits of elements from vector \a x. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type signbit(const simd<T, Abi> &x) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::signbit(x[i]);
		return {result.data(), element_aligned};
	}

	/** @copydoc fpclassify
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto fpclassify(const simd<T, Abi> &x) noexcept { return fpclassify(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc isfinite
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isfinite(const simd<T, Abi> &x) noexcept { return isfinite(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc isinf
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isinf(const simd<T, Abi> &x) noexcept { return isinf(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc isnan
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isnan(const simd<T, Abi> &x) noexcept { return isnan(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc isnormal
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isnormal(const simd<T, Abi> &x) noexcept { return isnormal(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }
	/** @copydoc signbit
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto signbit(const simd<T, Abi> &x) noexcept { return signbit(rebind_simd_t<detail::promote_t<T>, simd<T, Abi>>{x}); }

	/** Determines if elements of \a a are greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isgreater(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isgreater(a[i], b[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a a are greater than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isgreaterequal(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isgreaterequal(a[i], b[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a a are less than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isless(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isless(a[i], b[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a a are less than or equal to elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type islessequal(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::islessequal(a[i], b[i]);
		return {result.data(), element_aligned};
	}
	/** Determines if elements of \a a are less than or greater than elements of \a b without setting floating-point exceptions. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type islessgreater(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::islessgreater(a[i], b[i]);
		return {result.data(), element_aligned};
	}
	/** Determines is either elements of \a a or \a b are unordered. */
	template<std::floating_point T, typename Abi>
	[[nodiscard]] constexpr typename simd<T, Abi>::mask_type isunordered(const simd<T, Abi> &a, const simd<T, Abi> &b) noexcept
	{
		std::array<bool, simd_size_v<T, Abi>> result = {};
		for (std::size_t i = 0; i < simd<T, Abi>::size(); ++i)
			result[i] = std::isunordered(a[i], b[i]);
		return {result.data(), element_aligned};
	}

	/** @copydoc isgreater
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isgreater(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return isgreater(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc isgreaterequal
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isgreaterequal(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return isgreaterequal(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc isless
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isless(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return isless(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc islessequal
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto islessequal(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return islessequal(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc islessgreater
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto islessgreater(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return islessgreater(promoted_t{a}, promoted_t{b});
	}
	/** @copydoc isunordered
	 * @note Arguments are promoted to `double`, or `long double` if one of the arguments is `long double`. */
	template<typename T0, typename T1, typename Abi>
	[[nodiscard]] constexpr DPM_FORCEINLINE auto isunordered(const simd<T0, Abi> &a, const simd<T1, Abi> &b) noexcept
	{
		using promoted_t = rebind_simd_t<detail::promote_t<T0, T1>, simd<T0, Abi>>;
		return isunordered(promoted_t{a}, promoted_t{b});
	}
#pragma endregion
}
