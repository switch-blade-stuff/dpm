/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "type.hpp"

#ifdef DPM_HAS_SSE2

#ifdef DPM_HANDLE_ERRORS
#ifndef DPM_USE_IMPORT

#include <cerrno>
#include <cfenv>

#endif
#endif

#include "../dispatch.hpp"
#include "fmadd.hpp"
#include "class.hpp"
#include "cpuid.hpp"

#define DPM_SINCOS_SIGN(tgt) DPM_PURE DPM_VECTORCALL DPM_TARGET(tgt)

namespace dpm
{
	namespace detail
	{
		enum sincos_op
		{
			OP_SINCOS = 3,
			OP_COS = 2,
			OP_SIN = 1,
		};

		template<typename T, sincos_op Mask>
		struct sincos_ret { using type = T; };
		template<typename T>
		struct sincos_ret<T, sincos_op::OP_SINCOS> { using type = std::pair<T, T>; };
		template<typename T, sincos_op Mask>
		using sincos_ret_t = typename sincos_ret<T, Mask>::type;

		template<sincos_op Mask, typename V>
		DPM_FORCEINLINE static sincos_ret_t<V, Mask> return_sincos(V sin, V cos) noexcept
		{
			if constexpr (Mask == sincos_op::OP_SINCOS)
				return {sin, cos};
			else if constexpr (Mask == sincos_op::OP_SIN)
				return sin;
			else
				return cos;
		}

#ifdef DPM_HAS_AVX
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("avx2") sincos_avx2(__m256, __m256, __m256, __m256) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("avx2") sincos_avx2(__m256d, __m256d, __m256d, __m256d) noexcept;
#endif
#ifdef DPM_HAS_FMA
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("fma") sincos_fma(__m128, __m128, __m128, __m128) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("fma") sincos_fma(__m128d, __m128d, __m128d, __m128d) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("fma") sincos_fma(__m256, __m256, __m256, __m256) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("fma") sincos_fma(__m256d, __m256d, __m256d, __m256d) noexcept;
#endif
#ifdef DPM_HAS_AVX
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("avx") sincos_avx(__m256, __m256, __m256, __m256) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("avx") sincos_avx(__m256d, __m256d, __m256d, __m256d) noexcept;
#endif
#ifdef DPM_HAS_SSE4_1
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("sse4.1") sincos_sse4_1(__m128, __m128, __m128, __m128) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("sse4.1") sincos_sse4_1(__m128d, __m128d, __m128d, __m128d) noexcept;
#endif

		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("sse2") sincos_sse(__m128, __m128, __m128, __m128) noexcept;
		template<sincos_op>
		inline static auto DPM_SINCOS_SIGN("sse2") sincos_sse(__m128d, __m128d, __m128d, __m128d) noexcept;

		template<typename T, sincos_op Mask, typename V>
		DPM_FORCEINLINE static auto impl_sincos(V x) noexcept
		{
			constexpr auto extent_bits = movemask_bits_v<T> * sizeof(V) / sizeof(T);
			const auto abs_x = abs(x);

			/* Check for infinity, NaN & errors. */
#ifdef DPM_PROPAGATE_NAN
			const auto nan = fill<V>(std::numeric_limits<T>::quiet_NaN());
			auto nan_mask = isnan(x);

#ifdef DPM_HANDLE_ERRORS
			const auto inf = fill<V>(std::numeric_limits<T>::infinity());
			const auto inf_mask = cmp_eq<T>(abs_x, inf);
			nan_mask = bit_or(nan_mask, inf_mask);

			if (movemask<T>(inf_mask)) [[unlikely]]
			{
				std::feraiseexcept(FE_INVALID);
				errno = EDOM;
			}
#endif
			if (movemask<T>(nan_mask) == fill_bits<extent_bits>()) [[unlikely]]
				return return_sincos<Mask>(nan, nan);
#else
			const auto nan_mask = undefined<V>();
#endif
#ifdef DPM_HANDLE_ERRORS
			const auto zero_mask = cmp_eq<T>(abs_x, setzero<V>());
			if (movemask<T>(zero_mask) == fill_bits<extent_bits>()) [[unlikely]]
				return return_sincos<Mask>(x, fill<V>(T{1.0}));
#else
			const auto zero_mask = undefined<V>();
#endif

#if !defined(DPM_HAS_FMA) && defined(DPM_DYNAMIC_DISPATCH)
			constinit static dispatcher sincos_disp = []() -> sincos_ret_t<V, Mask> (*)(V, V, V, V)
			{
				if (cpuid::has_fma())
					return sincos_fma<Mask>;
				else if constexpr(sizeof(V) == 16)
				{
#ifndef DPM_HAS_SSE4_1
					if (!cpuid::has_sse4_1()) return sincos_sse<Mask>;
#endif
					return sincos_sse4_1<Mask>;
				}
				else
				{
#ifndef DPM_HAS_AVX2
					if (!cpuid::has_avx2()) return sincos_avx<Mask>;
#endif
					return sincos_avx2<Mask>;
				}
			};
			return sincos_disp(x, abs_x, nan_mask, zero_mask);
#elif defined(DPM_HAS_FMA)
			return sincos_fma<Mask>(x, abs_x, nan_mask, zero_mask);
#else
			if constexpr (sizeof(V) == 16)
			{
#ifdef DPM_HAS_SSE4_1
				return sincos_sse4_1<Mask>(x, abs_x, nan_mask, zero_mask);
#else
				return sincos_sse<Mask>(x, abs_x, nan_mask, zero_mask);
#endif
			}
			else if constexpr (sizeof(V) == 32)
			{
#ifdef DPM_HAS_AVX2
				return sincos_avx2<Mask>(x, abs_x, nan_mask, zero_mask);
#else
				return sincos_avx<Mask>(x, abs_x, nan_mask, zero_mask);
#endif
			}
#endif
		}

		[[nodiscard]] std::pair<__m128, __m128> DPM_PUBLIC DPM_SINCOS_SIGN("sse2") sincos(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_SINCOS_SIGN("sse2") sin(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_SINCOS_SIGN("sse2") cos(__m128 x) noexcept;

		[[nodiscard]] std::pair<__m128d, __m128d> DPM_PUBLIC DPM_SINCOS_SIGN("sse2") sincos(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_SINCOS_SIGN("sse2") sin(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_SINCOS_SIGN("sse2") cos(__m128d x) noexcept;

#ifdef DPM_HAS_AVX
		[[nodiscard]] std::pair<__m256, __m256> DPM_PUBLIC DPM_SINCOS_SIGN("avx") sincos(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_SINCOS_SIGN("avx") sin(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_SINCOS_SIGN("avx") cos(__m256 x) noexcept;

		[[nodiscard]] std::pair<__m256d, __m256d> DPM_PUBLIC DPM_SINCOS_SIGN("avx") sincos(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_SINCOS_SIGN("avx") sin(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_SINCOS_SIGN("avx") cos(__m256d x) noexcept;
#endif
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