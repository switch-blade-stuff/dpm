/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

#include "../generic/type.hpp"
#include "../utility.hpp"

#include "abi.hpp"

#if defined(SVM_ARCH_X86) && (defined(SVM_HAS_SSE) || defined(SVM_DYNAMIC_DISPATCH))

#ifndef SVM_USE_IMPORT

#include <bit>

#endif

namespace svm
{
	namespace detail
	{
		using simd_abi::detail::has_x86_default;
		using simd_abi::detail::default_x86_align;

		template<typename T, std::size_t N, std::size_t A, typename V>
		concept x86_simd_overload = vectorizable<T> && (A == alignof(V) || (A == 0 && has_x86_default<T, N> && default_x86_align<T, N>::value == alignof(V)));
		template<typename T, std::size_t N, std::size_t A>
		concept x86_sse_overload = x86_simd_overload<T, N, A, __m128>;
		template<typename T, std::size_t N, std::size_t A>
		concept x86_avx_overload = x86_simd_overload<T, N, A, __m256>;
		template<typename T, std::size_t N, std::size_t A>
		concept x86_avx512_overload = x86_simd_overload<T, N, A, __m512>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_sse_overload<float, N, Align>
	class simd_mask<float, simd_abi::aligned_vector<N, Align>>
	{
		using vector_type = __m128;

		constexpr static auto data_size = detail::vector_array_size<float, N, vector_type>();
		using data_type = vector_type[data_size];

	public:
		using value_type = bool;
		using reference = detail::mask_reference<std::uint32_t, 0xffff'ffff>;

		using abi_type = simd_abi::aligned_vector<N, Align>;
		using simd_type = simd<float, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		static void assert_subscript([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size(), "simd_mask<T, simd_abi::sse<T>> subscript out of range");
		}

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the SIMD mask object with a native SSE vector. */
		constexpr simd_mask(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128)`. */
		constexpr simd_mask(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		constexpr simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_ps(std::bit_cast<float>(0xffff'ffff)) : _mm_setzero_ps();
			std::fill_n(m_data, data_size, v);
		}

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			assert_subscript(i);
			return reference{reinterpret_cast<std::uint32_t *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			assert_subscript(i);
			return reinterpret_cast<const std::uint32_t *>(m_data)[i];
		}

		[[nodiscard]] simd_mask operator!() const noexcept
		{
			simd_mask result;
			const auto mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(m_data[i], mask);
			return result;
		}

		[[nodiscard]] friend simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_and_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_or_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_and_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_or_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_xor_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_and_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_or_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		[[nodiscard]] friend simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
#ifdef SVM_HAS_SSE2
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = _mm_castps_si128(a.m_data[i]);
				const auto vb = _mm_castps_si128(b.m_data[i]);
				result.m_data[i] = _mm_castsi128_ps(_mm_cmpeq_epi32(va, vb));
			}
#else
			const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = _mm_and_ps(a.m_data[i], nan_mask);
				const auto vb = _mm_and_ps(b.m_data[i], nan_mask);
				result.m_data[i] = _mm_cmpeq_ps(va, vb);
			}
#endif
			return result;
		}
		[[nodiscard]] friend simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
#ifdef SVM_HAS_SSE2
			const auto inv_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = _mm_castps_si128(a.m_data[i]);
				const auto vb = _mm_castps_si128(b.m_data[i]);
				result.m_data[i] = _mm_xor_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(va, vb)), inv_mask);
			}
#else
			const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
			{
				const auto va = _mm_and_ps(a.m_data[i], nan_mask);
				const auto vb = _mm_and_ps(b.m_data[i], nan_mask);
				result.m_data[i] = _mm_cmpneq_ps(va, vb);
			}
#endif
			return result;
		}

		/** @warning Internal use only! */
		[[nodiscard]] bool _impl_all_of() const noexcept
		{
#ifdef SVM_HAS_SSE4_1
			if constexpr (data_size == 1)
			{
				const auto vi = _mm_castps_si128(m_data[0]);
				return !_mm_testz_si128(vi, vi);
			}
#endif
			auto result = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
				result = _mm_and_ps(result, m_data[i]);
			return _mm_movemask_ps(result) == 0b1111;
		}
		/** @warning Internal use only! */
		[[nodiscard]] bool _impl_any_of() const noexcept
		{
			auto result = _mm_setzero_ps();
			for (std::size_t i = 0; i < data_size; ++i)
				result = _mm_or_ps(result, m_data[i]);
#ifdef SVM_HAS_SSE4_1
			const auto vi = _mm_castps_si128(result);
			return !_mm_testz_si128(vi, vi);
#else
			return _mm_movemask_ps(result);
#endif
		}
		/** @warning Internal use only! */
		[[nodiscard]] bool _impl_none_of() const noexcept
		{
			auto result = _mm_setzero_ps();
			for (std::size_t i = 0; i < data_size; ++i)
				result = _mm_or_ps(result, m_data[i]);
#ifdef SVM_HAS_SSE4_1
			const auto vi = _mm_castps_si128(result);
			return _mm_testz_si128(vi, vi);
#else
			return !_mm_movemask_ps(result);
#endif
		}
		/** @warning Internal use only! */
		[[nodiscard]] bool _impl_some_of() const noexcept
		{
			auto any_mask = _mm_setzero_ps(), all_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
			for (std::size_t i = 0; i < data_size; ++i)
			{
				all_mask = _mm_and_ps(all_mask, m_data[i]);
				any_mask = _mm_or_ps(any_mask, m_data[i]);
			}
#ifdef SVM_HAS_SSE4_1
			const auto any_vi = _mm_castps_si128(any_mask);
			const auto all_vi = _mm_castps_si128(all_mask);
			return !_mm_testz_si128(any_vi, all_vi) && _mm_testz_si128(all_vi, all_vi);
#else
			return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
#endif
		}

		[[nodiscard]] std::size_t _impl_popcount() const noexcept
		{
			std::size_t result = 0, i = 0;
			while (i + 8 <= data_size)
			{
				std::uint32_t bits = 0;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 28;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 24;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 20;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 16;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 12;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 8;
				bits |= _mm_movemask_ps(m_data[(i++)]) << 4;
				bits |= _mm_movemask_ps(m_data[(i++)]);
				result += std::popcount(bits);
			}
			switch (std::uint32_t bits = 0; data_size - i)
			{
				case 7: bits |= _mm_movemask_ps(m_data[i++]) << 24;
				case 6: bits |= _mm_movemask_ps(m_data[i++]) << 20;
				case 5: bits |= _mm_movemask_ps(m_data[i++]) << 16;
				case 4: bits |= _mm_movemask_ps(m_data[i++]) << 12;
				case 3: bits |= _mm_movemask_ps(m_data[i++]) << 8;
				case 2: bits |= _mm_movemask_ps(m_data[i++]) << 4;
				case 1: result += std::popcount(bits | _mm_movemask_ps(m_data[i++]));
			}
			return result;
		}
		[[nodiscard]] std::size_t _impl_find_first_set() const noexcept
		{
			for (std::size_t i = 0; i < data_size; i += 8)
				switch (std::uint32_t bits = 0; data_size - i)
				{
					default: bits = _mm_movemask_ps(m_data[i + 7]) << 28;
					case 7: bits |= _mm_movemask_ps(m_data[i + 6]) << 24;
					case 6: bits |= _mm_movemask_ps(m_data[i + 5]) << 20;
					case 5: bits |= _mm_movemask_ps(m_data[i + 4]) << 16;
					case 4: bits |= _mm_movemask_ps(m_data[i + 3]) << 12;
					case 3: bits |= _mm_movemask_ps(m_data[i + 2]) << 8;
					case 2: bits |= _mm_movemask_ps(m_data[i + 1]) << 4;
					case 1: bits |= _mm_movemask_ps(m_data[i]);
					case 0: if (bits) return std::countr_zero(bits) + i * 4;
				}
			return 0;
		}
		[[nodiscard]] std::size_t _impl_find_last_set() const noexcept
		{
			for (std::size_t i = data_size; i != 0;)
			{
				std::uint32_t bits = 0, k = (i - 1) % 8;
				switch (k)
				{
					case 7: bits = _mm_movemask_ps(m_data[i - 8]);
					case 6: bits |= _mm_movemask_ps(m_data[i - 7]) << 4;
					case 5: bits |= _mm_movemask_ps(m_data[i - 6]) << 8;
					case 4: bits |= _mm_movemask_ps(m_data[i - 5]) << 12;
					case 3: bits |= _mm_movemask_ps(m_data[i - 4]) << 16;
					case 2: bits |= _mm_movemask_ps(m_data[i - 3]) << 20;
					case 1: bits |= _mm_movemask_ps(m_data[i - 2]) << 24;
					case 0: bits |= _mm_movemask_ps(m_data[i - 1]) << 28;
						break;
					default: SVM_UNREACHABLE();
				}
				if (bits) return (i * 4 - 1) - std::countl_zero(bits);
				i -= k + 1;
			}
			return 0;
		}

	private:
		data_type m_data;
	};

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool all_of(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_all_of();
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_any_of();
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_none_of();
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_some_of();
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask)
	noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_popcount();
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask)
	noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_find_first_set();
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<float, simd_abi::aligned_vector<N, A>> &mask)
	noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return mask._impl_find_last_set();
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_simd_overload<float, N, Align, __m128>
	class simd<float, simd_abi::aligned_vector<N, Align>>
	{
		using vector_type = __m128;

		constexpr static auto data_size = detail::vector_array_size<float, N, vector_type>();
		using data_type = vector_type[data_size];

	public:
		using value_type = float;
		using reference = detail::simd_reference<value_type>;

		using abi_type = simd_abi::aligned_vector<N, Align>;
		using mask_type = simd_mask<float, abi_type>;

		/** Returns width of the SIMD type. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		static void assert_subscript([[maybe_unused]] std::size_t i) noexcept
		{
			SVM_ASSERT(i < size(), "simd<T, simd_abi::sse<T>> subscript out of range");
		}

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the SIMD object with a native SSE vector. */
		constexpr simd(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD object with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128)`. */
		constexpr simd(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying SSE vector with \a value. */
		template<detail::compatible_element<value_type> U>
		simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_ps(static_cast<float>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying scalar with a value provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
		{
			detail::generate<data_size>(m_data, [&gen]<std::size_t I>(std::integral_constant<std::size_t, I>)
			{
				constexpr auto value_idx = I * sizeof(__m128);
				const auto f0 = gen(std::integral_constant<std::size_t, value_idx>());
				const auto f1 = value_idx + 1 < size() ? gen(std::integral_constant<std::size_t, value_idx + 1>()) : 0;
				const auto f2 = value_idx + 2 < size() ? gen(std::integral_constant<std::size_t, value_idx + 2>()) : 0;
				const auto f3 = value_idx + 3 < size() ? gen(std::integral_constant<std::size_t, value_idx + 3>()) : 0;
				return _mm_set_ps(f0, f1, f2, f3);
			});
		}

		simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		inline simd &operator++() noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_add_ps(m_data[i], _mm_set1_ps(1.0f));
			return *this;
		}
		inline simd &operator--() noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) m_data[i] = _mm_sub_ps(m_data[i], _mm_set1_ps(1.0f));
			return *this;
		}

		[[nodiscard]] simd operator+() const noexcept { return *this; }
		[[nodiscard]] simd operator-() const noexcept
		{
			simd result;
			const auto mask = _mm_set1_ps(std::bit_cast<float>(0x8000'0000));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(m_data[i], mask);
			return result;
		}

		friend mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		friend mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpneq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}

	private:
		data_type m_data;
	};
}

#endif