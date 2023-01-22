/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "m128/type.hpp"
#include "m256/type.hpp"
#include "m512/type.hpp"

namespace dpm
{
	namespace detail
	{
		template<typename To, typename ToAbi, typename From, typename FromAbi>
		inline void DPM_FORCEINLINE cast_impl(simd<To, ToAbi> &to, const simd<From, FromAbi> &from) noexcept
		{
			const auto from_data = reinterpret_cast<const From *>(ext::to_native_data(from).data());
			constexpr auto from_align = alignof(decltype(from));
			constexpr auto to_align = alignof(decltype(to));

			if constexpr (to_align > from_align)
				to.copy_from(from_data, overaligned<to_align>);
			else if constexpr (to_align < from_align)
				to.copy_from(from_data, element_aligned);
			else
				to.copy_from(from_data, vector_aligned);
		}

		template<std::size_t I, typename T, typename OutAbi, typename XAbi, typename... Abis>
		inline void DPM_FORCEINLINE concat_impl(simd_mask<T, OutAbi> &out, const simd_mask<T, XAbi> &x, const simd_mask<T, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<T *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(__m128) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd_mask<T, XAbi>::size()>(out, rest...);
		}
		template<std::size_t I, typename T, typename OutAbi, typename XAbi, typename... Abis>
		inline void DPM_FORCEINLINE concat_impl(simd<T, OutAbi> &out, const simd<T, XAbi> &x, const simd<T, Abis> &...rest) noexcept
		{
			auto *data = reinterpret_cast<T *>(ext::to_native_data(out).data());
			if constexpr (I % sizeof(__m128) != 0)
				x.copy_to(data + I, element_aligned);
			else
				x.copy_to(data + I, vector_aligned);

			if constexpr (sizeof...(Abis) != 0) concat_impl<I + simd<T, XAbi>::size()>(out, rest...);
		}
	}

	/** Implicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline auto simd_cast(const simd<U, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, U, detail::avec<N, A>> &&
	          detail::x86_overload_any<To, N, A> &&
	          detail::x86_overload_any<U, N, A>)
	{
		detail::cast_return_t<T, U, detail::avec<N, A>, simd<U, detail::avec<N, A>>::size()> result;
		detail::cast_impl(result, x);
		return result;
	}
	/** Explicitly converts elements of SIMD vector \a x to the `To` type, where `To` is either `typename T::value_type` or `T` if `T` is a scalar. */
	template<typename T, typename U, std::size_t N, std::size_t A, typename To = typename detail::deduce_cast<T>::type>
	[[nodiscard]] inline auto static_simd_cast(const simd<U, detail::avec<N, A>> &x) noexcept
	requires (detail::valid_simd_cast<T, U, detail::avec<N, A>> &&
	          detail::x86_overload_any<To, N, A> &&
	          detail::x86_overload_any<U, N, A>)
	{
		detail::static_cast_return_t<T, U, detail::avec<N, A>, simd<U, detail::avec<N, A>>::size()> result;
		detail::cast_impl(result, x);
		return result;
	}

	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, typename... Abis>
	[[nodiscard]] inline auto concat(const simd_mask<T, Abis> &...values) noexcept requires ((detail::x86_simd_abi_any<Abis, T> && ...))
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd_mask<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>> result;
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD vector. */
	template<typename T, typename... Abis>
	[[nodiscard]] inline auto concat(const simd<T, Abis> &...values) noexcept requires ((detail::x86_simd_abi_any<Abis, T> && ...))
	{
		if constexpr (sizeof...(values) == 1)
			return (values, ...);
		else
		{
			simd<T, simd_abi::deduce_t<T, (simd_size_v<T, Abis> + ...)>> result;
			detail::concat_impl<0>(result, values...);
			return result;
		}
	}

	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] inline auto concat(const std::array<simd_mask<T, detail::avec<N, A>>, M> &values) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
	{
		using result_t = simd_mask<T, detail::avec<N * M, A>>;
		if constexpr (M == 1)
			return values[0];
		else
		{
			result_t result;
			for (std::size_t i = 0; i < M; ++i)
			{
				auto *data = reinterpret_cast<T *>(ext::to_native_data(result).data());
				if ((i * N) % sizeof(ext::native_data_type_t<result_t>) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}
	/** Concatenates elements of \a values into a single SIMD mask. */
	template<typename T, std::size_t N, std::size_t A, std::size_t M>
	[[nodiscard]] inline auto concat(const std::array<simd<T, detail::avec<N, A>>, M> &values) noexcept requires detail::x86_overload_any<T, N, A> && detail::x86_overload_any<T, M, A>
	{
		using result_t = simd<T, detail::avec<N * M, A>>;
		if constexpr (M == 1)
			return values[0];
		else
		{
			result_t result;
			for (std::size_t i = 0; i < M; ++i)
			{
				auto *data = reinterpret_cast<T *>(ext::to_native_data(result).data());
				if ((i * N) % sizeof(ext::native_data_type_t<result_t>) != 0)
					values[i].copy_to(data + i * N, element_aligned);
				else
					values[i].copy_to(data + i * N, vector_aligned);
			}
			return result;
		}
	}

	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask a copy of the `i + j * V::size()`th element from \a x.
	 * @note Size of \a x must be a multiple of `V::size()`. */
	template<typename V, std::size_t N, std::size_t A, typename U = typename V::simd_type::value_type>
	[[nodiscard]] inline auto split(const simd_mask<U, detail::avec<N, A>> &x) noexcept requires detail::can_split_mask<V, detail::avec<N, A>> && detail::x86_overload_any<U, N, A>
	{
		std::array<V, simd_size_v<U, detail::avec<N, A>> / V::size()> result;
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			const auto *data = reinterpret_cast<const U *>(ext::to_native_data(x).data());
			result[j].copy_from(data + j * V::size(), element_aligned);
		}
		return result;
	}
	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * V::size()`th element from \a x.
	 * @note Size of \a x must be a multiple of `V::size()`. */
	template<typename V, std::size_t N, std::size_t A, typename U = typename V::simd_type::value_type>
	[[nodiscard]] inline auto split(const simd<U, detail::avec<N, A>> &x) noexcept requires detail::can_split_mask<V, detail::avec<N, A>> && detail::x86_overload_any<U, N, A>
	{
		std::array<V, simd_size_v<U, detail::avec<N, A>> / V::size()> result;
		for (std::size_t j = 0; j < result.size(); ++j)
		{
			const auto *data = reinterpret_cast<const U *>(ext::to_native_data(x).data());
			result[j].copy_from(data + j * V::size(), element_aligned);
		}
		return result;
	}

	/** Returns an array of SIMD masks where every `i`th element of the `j`th mask is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a x. */
	template<std::size_t N, typename T, std::size_t M, std::size_t A>
	[[nodiscard]] inline auto split_by(const simd_mask<T, detail::avec<M, A>> &x) noexcept requires (M % N == 0 && detail::x86_overload_any<T, M, A>)
	{
		return split<resize_simd_t<M / N, simd_mask<T, detail::avec<M, A>>>>(x);
	}
	/** Returns an array of SIMD vectors where every `i`th element of the `j`th vector is a copy of the `i + j * (simd_size_v<T, Abi> / N)`th element from \a x. */
	template<std::size_t N, typename T, std::size_t M, std::size_t A>
	[[nodiscard]] inline auto split_by(const simd<T, detail::avec<M, A>> &x) noexcept requires (M % N == 0 && detail::x86_overload_any<T, M, A>)
	{
		return split<resize_simd_t<M / N, simd<T, detail::avec<M, A>>>>(x);
	}
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif