/*
 * Created by switchblade on 2023-02-05.
 */

#pragma once

#include "../fconst.hpp"
#include "mbase.hpp"

namespace dpm
{
	namespace detail
	{
		template<typename T, typename V>
		[[nodiscard]] DPM_FORCEINLINE V masksign(V x) noexcept { return bit_and(x, fill<V>(sign_bit<T>)); }
		template<typename T, typename V>
		[[nodiscard]] DPM_FORCEINLINE V copysign(V x, V m) noexcept { return bit_or(abs<T>(x), m); }
	}

	/** Copies sign bit from elements of vector \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, const detail::x86_simd<T, N, A> &sign) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		detail::vectorize([](auto &res, auto x, auto s) { res = detail::copysign<T>(x, detail::masksign<T>(s)); }, result, x, sign);
		return result;
	}
	/** Copies sign bit from \a sign to elements of vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE simd<T, detail::avec<N, A>> copysign(const detail::x86_simd<T, N, A> &x, T sign) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		const auto sign_vec = detail::masksign<T>(detail::fill<ext::native_data_type_t<detail::x86_simd<T, N, A>>>(sign));
		detail::vectorize([s = sign_vec](auto &res, auto x) { res = detail::copysign<T>(x, s); }, result, x);
		return result;
	}
}