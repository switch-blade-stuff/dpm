/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "type.hpp"

#ifdef DPM_HAS_SSE2

#include "fmadd.hpp"
#include "class.hpp"

namespace dpm
{
	namespace detail
	{
		/* TODO: Implement single-precision sine & cosine. */
		[[nodiscard]] std::pair<__m128, __m128> DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") sincos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") sin(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") cos(__m128 x) noexcept;
		
		[[nodiscard]] std::pair<__m128d, __m128d> DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") sincos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") sin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") cos(__m128d x) noexcept;
	}

	/** Calculates sine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> sin(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			ext::to_native_data(result)[i] = detail::sin(ext::to_native_data(x)[i]);
		return result;
	}
	/** Calculates cosine of elements in vector \a x, and returns the resulting vector. */
	template<std::floating_point T, std::size_t N, std::size_t A>
	[[nodiscard]] DPM_FORCEINLINE detail::x86_simd<T, N, A> cos(const detail::x86_simd<T, N, A> &x) noexcept requires detail::x86_overload_any<T, N, A>
	{
		detail::x86_simd<T, N, A> result = {};
		for (std::size_t i = 0; i < ext::native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			ext::to_native_data(result)[i] = detail::cos(ext::to_native_data(x)[i]);
		return result;
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Calculates sine and cosine of elements in vector \a x, and assigns results to elements of \a out_sin and \a out_cos respectively. */
		template<std::floating_point T, std::size_t N, std::size_t A>
		DPM_FORCEINLINE void sincos(const detail::x86_simd<T, N, A> &x, detail::x86_simd<T, N, A> &out_sin, detail::x86_simd<T, N, A> &out_cos) noexcept requires detail::x86_overload_any<T, N, A>
		{
			for (std::size_t i = 0; i < native_data_size_v<detail::x86_simd<T, N, A>>; ++i)
			{
				const auto [sin, cos] = detail::sincos(to_native_data(x)[i]);
				to_native_data(out_sin)[i] = sin;
				to_native_data(out_cos)[i] = cos;
			}
		}
	}
}

#endif