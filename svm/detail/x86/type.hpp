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
		using simd_abi::detail::x86_sse_overload;
		using simd_abi::detail::x86_avx_overload;
		using simd_abi::detail::x86_avx512_overload;

		template<std::size_t N, std::size_t A>
		using avec = simd_abi::ext::aligned_vector<N, A>;
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
#ifdef SVM_HAS_SSE4_1
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<float, detail::avec<N, A>> blend(
				const simd_mask<float, detail::avec<N, A>> &a,
				const simd_mask<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_sse_overload<float, N, A>;
#endif

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline __m128 *to_native_data(simd_mask<float, detail::avec<N, A>> &value)
		noexcept requires detail::x86_sse_overload<float, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline const __m128 *to_native_data(const simd_mask<float, detail::avec<N, A>> &value)
		noexcept requires detail::x86_sse_overload<float, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_sse_overload<float, N, Align>
	class simd_mask<float, detail::avec<N, Align>>
	{
		using vector_type = __m128;

		constexpr static auto data_size = detail::align_vector_array<float, N, vector_type>();
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];

	public:
		using value_type = bool;
		using reference = detail::mask_reference<std::uint32_t>;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<float, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		friend struct detail::simd_access<simd_type>;
		friend struct detail::simd_access<simd_mask>;

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

		/** Initializes the SIMD mask object with a native SSE vector.
		 * @note This constructor is available for overload resolution only when the SIMD mask contains a single SSE vector. */
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
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		inline simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && (OtherAlign == 0 || OtherAlign >= alignment))
				std::copy_n(reinterpret_cast<const vector_type *>(ext::to_native_data(other)), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 4)
			{
				float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
				switch (size() - i)
				{
					default: f3 = std::bit_cast<float>(detail::extend_bool<std::uint32_t>(mem[i + 3]));
					case 3: f2 = std::bit_cast<float>(detail::extend_bool<std::uint32_t>(mem[i + 2]));
					case 2: f1 = std::bit_cast<float>(detail::extend_bool<std::uint32_t>(mem[i + 1]));
					case 1: f0 = std::bit_cast<float>(detail::extend_bool<std::uint32_t>(mem[i]));
				}
				m_data[i / 4] = _mm_set_ps(f3, f2, f1, f0);
			}
		}
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			for (std::size_t i = 0; i < size(); i += 4)
				switch (const auto bits = _mm_movemask_ps(m_data[i / 4]); size() - i)
				{
					default: mem[i + 3] = bits & 0b1000;
					case 3: mem[i + 2] = bits & 0b0100;
					case 2: mem[i + 1] = bits & 0b0010;
					case 1: mem[i] = bits & 0b0001;
				}
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

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_sse_overload<float, N, A>
		struct simd_access<simd_mask<float, avec<N, A>>>
		{
			using mask_t = simd_mask<float, avec<N, A>>;

			[[nodiscard]] static bool all_of(const mask_t &mask) noexcept
			{
#ifdef SVM_HAS_SSE4_1
				if constexpr (mask_t::data_size == 1)
				{
					const auto vi = _mm_castps_si128(mask.m_data[0]);
					return _mm_test_all_ones(vi);
				}
#endif
				auto result = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < mask_t::data_size; ++i)
					result = _mm_and_ps(result, mask.m_data[i]);
				return _mm_movemask_ps(result) == 0b1111;
			}
			[[nodiscard]] static bool any_of(const mask_t &mask) noexcept
			{
				auto result = _mm_setzero_ps();
				for (std::size_t i = 0; i < mask_t::data_size; ++i)
					result = _mm_or_ps(result, mask.m_data[i]);
#ifdef SVM_HAS_SSE4_1
				const auto vi = _mm_castps_si128(result);
				return !_mm_testz_si128(vi, vi);
#else
				return _mm_movemask_ps(result);
#endif
			}
			[[nodiscard]] static bool none_of(const mask_t &mask) noexcept
			{
				auto result = _mm_setzero_ps();
				for (std::size_t i = 0; i < mask_t::data_size; ++i)
					result = _mm_or_ps(result, mask.m_data[i]);
#ifdef SVM_HAS_SSE4_1
				const auto vi = _mm_castps_si128(result);
				return _mm_testz_si128(vi, vi);
#else
				return !_mm_movemask_ps(result);
#endif
			}
			[[nodiscard]] static bool some_of(const mask_t &mask) noexcept
			{
				auto any_mask = _mm_setzero_ps(), all_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < mask_t::data_size; ++i)
				{
					all_mask = _mm_and_ps(all_mask, mask.m_data[i]);
					any_mask = _mm_or_ps(any_mask, mask.m_data[i]);
				}
#ifdef SVM_HAS_SSE4_1
				const auto any_vi = _mm_castps_si128(any_mask);
				const auto all_vi = _mm_castps_si128(all_mask);
				return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
				return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
#endif
			}

			[[nodiscard]] static std::size_t popcount(const mask_t &mask) noexcept
			{
				std::size_t result = 0, i = 0;
				while (i + 8 <= mask_t::data_size)
				{
					std::uint32_t bits = 0;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 28;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 24;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 20;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 16;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 12;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 8;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]) << 4;
					bits |= _mm_movemask_ps(mask.m_data[(i++)]);
					result += std::popcount(bits);
				}
				switch (std::uint32_t bits = 0; mask_t::data_size - i)
				{
					case 7: bits |= _mm_movemask_ps(mask.m_data[i++]) << 24;
					case 6: bits |= _mm_movemask_ps(mask.m_data[i++]) << 20;
					case 5: bits |= _mm_movemask_ps(mask.m_data[i++]) << 16;
					case 4: bits |= _mm_movemask_ps(mask.m_data[i++]) << 12;
					case 3: bits |= _mm_movemask_ps(mask.m_data[i++]) << 8;
					case 2: bits |= _mm_movemask_ps(mask.m_data[i++]) << 4;
					case 1: result += std::popcount(bits | _mm_movemask_ps(mask.m_data[i++]));
				}
				return result;
			}
			[[nodiscard]] static std::size_t find_first_set(const mask_t &mask) noexcept
			{
				for (std::size_t i = 0; i < mask_t::data_size; i += 8)
					switch (std::uint32_t bits = 0; mask_t::data_size - i)
					{
						default: bits = _mm_movemask_ps(mask.m_data[i + 7]) << 28;
						case 7: bits |= _mm_movemask_ps(mask.m_data[i + 6]) << 24;
						case 6: bits |= _mm_movemask_ps(mask.m_data[i + 5]) << 20;
						case 5: bits |= _mm_movemask_ps(mask.m_data[i + 4]) << 16;
						case 4: bits |= _mm_movemask_ps(mask.m_data[i + 3]) << 12;
						case 3: bits |= _mm_movemask_ps(mask.m_data[i + 2]) << 8;
						case 2: bits |= _mm_movemask_ps(mask.m_data[i + 1]) << 4;
						case 1: bits |= _mm_movemask_ps(mask.m_data[i]);
						case 0: if (bits) return std::countr_zero(bits) + i * 4;
					}
				return 0;
			}
			[[nodiscard]] static std::size_t find_last_set(const mask_t &mask) noexcept
			{
				for (std::size_t i = mask_t::data_size, k; i != 0; i -= k + 1)
					switch (std::uint32_t bits = 0; k = ((i - 1) % 8))
					{
						case 7: bits = _mm_movemask_ps(mask.m_data[i - 8]);
						case 6: bits |= _mm_movemask_ps(mask.m_data[i - 7]) << 4;
						case 5: bits |= _mm_movemask_ps(mask.m_data[i - 6]) << 8;
						case 4: bits |= _mm_movemask_ps(mask.m_data[i - 5]) << 12;
						case 3: bits |= _mm_movemask_ps(mask.m_data[i - 4]) << 16;
						case 2: bits |= _mm_movemask_ps(mask.m_data[i - 3]) << 20;
						case 1: bits |= _mm_movemask_ps(mask.m_data[i - 2]) << 24;
						case 0: bits |= _mm_movemask_ps(mask.m_data[i - 1]) << 28;
							if (bits) return (i * 4 - 1) - std::countl_zero(bits);
							break;
						default: SVM_UNREACHABLE();
					}
				return 0;
			}

#ifdef SVM_HAS_SSE4_1
			[[nodiscard]] static mask_t blend(const mask_t &a, const mask_t &b, const mask_t &m) noexcept
			{
				mask_t result;
				for (std::size_t i = 0; i < mask_t::data_size; ++i)
				{
					const auto m_data = m.m_data[i];
					const auto b_data = b.m_data[i];
					const auto a_data = a.m_data[i];
					result.m_data[i] = _mm_blendv_ps(a_data, b_data, m_data);
				}
				return result;
			}
#endif

			[[nodiscard]] static auto *to_native_data(mask_t &value) noexcept { return value.m_data; }
			[[nodiscard]] static auto *to_native_data(const mask_t &value) noexcept { return value.m_data; }
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
#ifdef SVM_HAS_SSE4_1
		/** Replaces elements of mask \a a with selected elements of where expression \a b. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<float, detail::avec<N, A>> blend(
				const simd_mask<float, detail::avec<N, A>> &a,
				const simd_mask<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_sse_overload<float, N, A>
		{
			return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::blend(a, b, m);
		}
#endif

		/** Returns a pointer to the underlying native SSE vector (or array of SSE vectors) for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline __m128 *to_native_data(simd_mask<float, detail::avec<N, A>> &value)
		noexcept requires detail::x86_sse_overload<float, N, A>
		{
			return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a const pointer to the underlying native SSE vector (or array of SSE vectors) for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline const __m128 *to_native_data(const simd_mask<float, detail::avec<N, A>> &value)
		noexcept requires detail::x86_sse_overload<float, N, A>
		{
			return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::to_native_data(value);
		}
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool all_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::all_of(mask);
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::any_of(mask);
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::none_of(mask);
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::some_of(mask);
	}

	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::popcount(mask);
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::find_first_set(mask);
	}
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_sse_overload<float, N, A>
	{
		return detail::simd_access<simd_mask<float, detail::avec<N, A>>>::find_last_set(mask);
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_sse_overload<float, N, Align>
	class simd<float, detail::avec<N, Align>>
	{
		using vector_type = __m128;

		constexpr static auto data_size = detail::align_vector_array<float, N, vector_type>();
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];

	public:
		using value_type = float;
		using reference = detail::simd_reference<value_type>;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<float, abi_type>;

		/** Returns width of the SIMD type. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	private:
		friend struct detail::simd_access<simd>;
		friend struct detail::simd_access<mask_type>;

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

		/** Initializes the SIMD vector with a native SSE vector.
		 * @note This constructor is available for overload resolution only when the SIMD vector contains a single SSE vector. */
		constexpr simd(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128)`. */
		constexpr simd(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_ps(static_cast<float>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
		{
			detail::generate<data_size>(m_data, [&gen]<std::size_t I>(std::integral_constant<std::size_t, I>)
			{
				float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
				switch (constexpr auto value_idx = I * 4; size() - value_idx)
				{
					default: f3 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 3>());
					case 3: f2 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 2>());
					case 2: f1 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>());
					case 1: f0 = std::invoke(gen, std::integral_constant<std::size_t, value_idx>());
				}
				return _mm_set_ps(f3, f2, f1, f0);
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd(const simd<U, simd_abi::ext::aligned_vector<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (!std::same_as<U, value_type>)
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
			else if constexpr (OtherAlign == 0 || OtherAlign >= alignment)
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other)), vector_aligned);
			else if constexpr (OtherAlign != alignof(value_type))
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other)), overaligned<OtherAlign>);
			else
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other)), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (std::same_as<U, value_type> && (std::derived_from<Flags, vector_aligned_tag> || detail::overaligned_tag_value_v<Flags> >= alignment))
				std::copy_n(reinterpret_cast<const vector_type *>(mem), data_size, m_data);
			else
				std::copy_n(mem, size(), reinterpret_cast<value_type *>(m_data));
		}
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags>
		{
			if constexpr (std::same_as<U, value_type> && (std::derived_from<Flags, vector_aligned_tag> || detail::overaligned_tag_value_v<Flags> >= alignment))
				std::copy_n(m_data, data_size, reinterpret_cast<const vector_type *>(mem));
			else
				std::copy_n(reinterpret_cast<value_type *>(m_data), size(), mem);
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
			const auto sign_mask = _mm_set1_ps(std::bit_cast<float>(0x8000'0000));
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_xor_ps(m_data[i], sign_mask);
			return result;
		}

		[[nodiscard]] friend simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_add_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_sub_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend simd &operator+=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_add_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd &operator-=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_sub_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] friend simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_mul_ps(a.m_data[i], b.m_data[i]);
			return result;
		}
		[[nodiscard]] friend simd operator/(const simd &a, const simd &b) noexcept
		{
			simd result;
			for (std::size_t i = 0; i < data_size; ++i) result.m_data[i] = _mm_div_ps(a.m_data[i], b.m_data[i]);
			return result;
		}

		friend simd &operator*=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_mul_ps(a.m_data[i], b.m_data[i]);
			return a;
		}
		friend simd &operator/=(simd &a, const simd &b) noexcept
		{
			for (std::size_t i = 0; i < data_size; ++i) a.m_data[i] = _mm_div_ps(a.m_data[i], b.m_data[i]);
			return a;
		}

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			assert_subscript(i);
			return reference{reinterpret_cast<float *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			assert_subscript(i);
			return reinterpret_cast<const float *>(m_data)[i];
		}

		[[nodiscard]] friend mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpeq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpneq_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator<=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmple_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator>=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpge_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator<(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmplt_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator>(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			for (std::size_t i = 0; i < data_size; ++i) mask_data[i] = _mm_cmpgt_ps(a.m_data[i], b.m_data[i]);
			return {mask_data};
		}

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_sse_overload<float, N, A>
		struct simd_access<simd<float, avec<N, A>>>
		{
			using simd_t = simd<float, avec<N, A>>;
			using mask_t = simd_mask<float, avec<N, A>>;

#ifdef SVM_HAS_SSE4_1
			[[nodiscard]] static simd_t blend(const simd_t &a, const simd_t &b, const mask_t &m) noexcept
			{
				simd_t result;
				for (std::size_t i = 0; i < simd_t::data_size; ++i)
				{
					const auto m_data = m.m_data[i];
					const auto b_data = b.m_data[i];
					const auto a_data = a.m_data[i];
					result.m_data[i] = _mm_blendv_ps(a_data, b_data, m_data);
				}
				return result;
			}
#endif

			[[nodiscard]] static auto *to_native_data(simd_t &value) noexcept { return value.m_data; }
			[[nodiscard]] static auto *to_native_data(const simd_t &value) noexcept { return value.m_data; }
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
#ifdef SVM_HAS_SSE4_1
		/** Replaces elements of vector \a a with selected elements of where expression \a b. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<float, detail::avec<N, A>> blend(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_sse_overload<float, N, A>
		{
			return detail::simd_access<simd<float, detail::avec<N, A>>>::blend(a, b, m);
		}
#endif

		/** Returns a pointer to the underlying native SSE vector (or array of SSE vectors) for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline __m128 *to_native_data(simd<float, detail::avec<N, A>> &value)
		noexcept requires detail::x86_sse_overload<float, N, A>
		{
			return detail::simd_access<simd<float, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a const pointer to the underlying native SSE vector (or array of SSE vectors) for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline const __m128 *to_native_data(const simd<float, detail::avec<N, A>> &value)
		noexcept requires detail::x86_sse_overload<float, N, A>
		{
			return detail::simd_access<simd<float, detail::avec<N, A>>>::to_native_data(value);
		}
	}
}

#endif