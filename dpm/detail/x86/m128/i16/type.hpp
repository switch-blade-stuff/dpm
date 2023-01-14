/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../fwd.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm
{
	namespace detail
	{
		template<std::size_t N>
		[[nodiscard]] static __m128i x86_maskzero_vector_i16(__m128i v, std::size_t i) noexcept
		{
			switch ([[maybe_unused]] const auto mask = std::bit_cast<float>(0xffff'ffff); N - i)
			{
#ifdef DPM_HAS_SSE4_1
				case 7: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1000'0000);
				case 6: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1100'0000);
				case 5: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1110'0000);
				case 4: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1111'0000);
				case 3: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1111'1000);
				case 2: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1111'1100);
				case 1: return _mm_blend_epi16(v, _mm_setzero_si128(), 0b1111'1110);
#else
					case 7: return _mm_and_si128(v, _mm_set_epi16(0, mask, mask, mask, mask, mask, mask, mask));
					case 6: return _mm_and_si128(v, _mm_set_epi16(0, 0, mask, mask, mask, mask, mask, mask));
					case 5: return _mm_and_si128(v, _mm_set_epi16(0, 0, 0, mask, mask, mask, mask, mask));
					case 4: return _mm_and_si128(v, _mm_set_epi16(0, 0, 0, 0, mask, mask, mask, mask));
					case 3: return _mm_and_si128(v, _mm_set_epi16(0, 0, 0, 0, 0, mask, mask, mask));
					case 2: return _mm_and_si128(v, _mm_set_epi16(0, 0, 0, 0, 0, 0, mask, mask));
					case 1: return _mm_and_si128(v, _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, mask));
#endif
				default: return v;
			}
		}
		template<std::size_t N>
		[[nodiscard]] static __m128i x86_maskone_vector_i16(__m128i v, std::size_t i) noexcept
		{
			switch (const auto mask = std::bit_cast<std::int32_t>(0xffff'ffff); N - i)
			{
#ifdef DPM_HAS_SSE4_1
				case 7: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1000'0000);
				case 6: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1100'0000);
				case 5: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1110'0000);
				case 4: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1111'0000);
				case 3: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1111'1000);
				case 2: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1111'1100);
				case 1: return _mm_blend_epi16(v, _mm_set1_epi32(mask), 0b1111'1110);
#else
					case 7: return _mm_or_si128(v, _mm_set_epi16(mask, 0, 0, 0, 0, 0, 0, 0));
					case 6: return _mm_or_si128(v, _mm_set_epi16(mask, mask, 0, 0, 0, 0, 0, 0));
					case 5: return _mm_or_si128(v, _mm_set_epi16(mask, mask, mask, 0, 0, 0, 0, 0));
					case 4: return _mm_or_si128(v, _mm_set_epi16(mask, mask, mask, mask, 0, 0, 0, 0));
					case 3: return _mm_or_si128(v, _mm_set_epi16(mask, mask, mask, mask, mask, 0, 0, 0));
					case 2: return _mm_or_si128(v, _mm_set_epi16(mask, mask, mask, mask, mask, mask, 0, 0));
					case 1: return _mm_or_si128(v, _mm_set_epi16(mask, mask, mask, mask, mask, mask, mask, 0));
#endif
				default: return v;
			}
		}

		template<integral_of_size<2> I, std::size_t N>
		struct x86_mask_impl<I, __m128i, N>
		{
			template<typename U>
			static U &data_at(__m128i *data, std::size_t i) noexcept { return reinterpret_cast<U *>(data)[i]; }
			template<typename U>
			static std::add_const_t<U> &data_at(const __m128i *data, std::size_t i) noexcept { return reinterpret_cast<std::add_const_t<U> *>(data)[i]; }

			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_from(const bool *src, __m128i *dst, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 8)
				{
					I i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
					switch (N - i)
					{
						default: i7 = extend_bool<std::int16_t>(src[i + 7]); [[fallthrough]];
						case 7: i6 = extend_bool<std::int16_t>(src[i + 6]); [[fallthrough]];
						case 6: i5 = extend_bool<std::int16_t>(src[i + 5]); [[fallthrough]];
						case 5: i4 = extend_bool<std::int16_t>(src[i + 4]); [[fallthrough]];
						case 4: i3 = extend_bool<std::int16_t>(src[i + 3]); [[fallthrough]];
						case 3: i2 = extend_bool<std::int16_t>(src[i + 2]); [[fallthrough]];
						case 2: i1 = extend_bool<std::int16_t>(src[i + 1]); [[fallthrough]];
						case 1: i0 = extend_bool<std::int16_t>(src[i]);
					}
					dst[i / 8] = _mm_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0);
				}
			}
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_to(bool *dst, const __m128i *src, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 8)
				{
					switch (const auto bits = _mm_movemask_epi8(src[i / 8]); N - i)
					{
						default: dst[i + 7] = bits & 0b1100'0000'0000'0000; [[fallthrough]];
						case 7: dst[i + 6] = bits & 0b0011'0000'0000'0000; [[fallthrough]];
						case 6: dst[i + 5] = bits & 0b0000'1100'0000'0000; [[fallthrough]];
						case 5: dst[i + 4] = bits & 0b0000'0011'0000'0000; [[fallthrough]];
						case 4: dst[i + 3] = bits & 0b0000'0000'1100'0000; [[fallthrough]];
						case 3: dst[i + 2] = bits & 0b0000'0000'0011'0000; [[fallthrough]];
						case 2: dst[i + 1] = bits & 0b0000'0000'0000'1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0000'0000'0000'0011;
					}
				}
			}

			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_from(const bool *src, __m128i *dst, const __m128i *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 8)
				{
					I i0 = 0, i1 = 0, i2 = 0, i3 = 0, i4 = 0, i5 = 0, i6 = 0, i7 = 0;
					switch (N - i)
					{
						default: i7 = extend_bool<std::int16_t>(src[i + 7]); [[fallthrough]];
						case 7: i6 = extend_bool<std::int16_t>(src[i + 6]); [[fallthrough]];
						case 6: i5 = extend_bool<std::int16_t>(src[i + 5]); [[fallthrough]];
						case 5: i4 = extend_bool<std::int16_t>(src[i + 4]); [[fallthrough]];
						case 4: i3 = extend_bool<std::int16_t>(src[i + 3]); [[fallthrough]];
						case 3: i2 = extend_bool<std::int16_t>(src[i + 2]); [[fallthrough]];
						case 2: i1 = extend_bool<std::int16_t>(src[i + 1]); [[fallthrough]];
						case 1: i0 = extend_bool<std::int16_t>(src[i]);
					}
					dst[i / 8] = _mm_and_si128(_mm_set_epi16(i7, i6, i5, i4, i3, i2, i1, i0), mask[i / 8]);
				}
			}
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_to(bool *dst, const __m128i *src, const __m128i *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 8)
				{
					const auto bits = _mm_movemask_epi8(_mm_and_si128(src[i / 8], mask[i / 8]));
					switch (N - i)
					{
						default: dst[i + 7] = bits & 0b1100'0000'0000'0000; [[fallthrough]];
						case 7: dst[i + 6] = bits & 0b0011'0000'0000'0000; [[fallthrough]];
						case 6: dst[i + 5] = bits & 0b0000'1100'0000'0000; [[fallthrough]];
						case 5: dst[i + 4] = bits & 0b0000'0011'0000'0000; [[fallthrough]];
						case 4: dst[i + 3] = bits & 0b0000'0000'1100'0000; [[fallthrough]];
						case 3: dst[i + 2] = bits & 0b0000'0000'0011'0000; [[fallthrough]];
						case 2: dst[i + 1] = bits & 0b0000'0000'0000'1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0000'0000'0000'0011;
					}
				}
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void invert(__m128i *dst, const __m128i *src) noexcept
			{
				const auto mask = _mm_set1_epi16(static_cast<std::int16_t>(0xffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_si128(src[i], mask);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_and(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_and_si128(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_or(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_or_si128(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_xor(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_si128(a[i], b[i]);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_eq(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpeq_epi16(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_ne(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				const auto inv_mask = _mm_set1_epi16(static_cast<std::int16_t>(0xffff));
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_si128(_mm_cmpeq_epi16(a[i], b[i]), inv_mask);
			}

			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool all_of(const __m128i *mask) noexcept
			{
#ifdef DPM_HAS_SSE4_1
				if constexpr (M == 1) return _mm_test_all_ones(x86_maskone_vector_i16<N>(mask[0], 0));
#endif
				auto result = _mm_set1_epi16(static_cast<std::int16_t>(0xffff));
				for (std::size_t i = 0; i < M; ++i) result = _mm_and_si128(result, x86_maskone_vector_i16<N>(mask[i], i * 8));
				return _mm_movemask_epi8(result) == 0xffff;
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool any_of(const __m128i *mask) noexcept
			{
				auto result = _mm_setzero_si128();
				for (std::size_t i = 0; i < M; ++i) result = _mm_or_si128(result, x86_maskzero_vector_i16<N>(mask[i], i * 8));
#ifdef DPM_HAS_SSE4_1
				return !_mm_testz_si128(result, result);
#else
				return _mm_movemask_epi8(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool none_of(const __m128i *mask) noexcept
			{
				auto result = _mm_setzero_si128();
				for (std::size_t i = 0; i < M; ++i) result = _mm_or_si128(result, x86_maskzero_vector_i16<N>(mask[i], i * 8));
#ifdef DPM_HAS_SSE4_1
				return _mm_testz_si128(result, result);
#else
				return !_mm_movemask_epi8(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool some_of(const __m128i *mask) noexcept
			{
				auto any_mask = _mm_setzero_si128(), all_mask = _mm_set1_epi16(static_cast<std::int16_t>(0xffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					all_mask = _mm_and_si128(all_mask, x86_maskzero_vector_i16<N>(mask[i], i * 8));
					any_mask = _mm_or_si128(any_mask, x86_maskone_vector_i16<N>(mask[i], i * 8));
				}
#ifdef DPM_HAS_SSE4_1
				return !_mm_testz_si128(any_mask, any_mask) && !_mm_test_all_ones(all_mask);
#else
				return _mm_movemask_epi8(any_mask) && _mm_movemask_epi8(all_mask) != 0xffff;
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t popcount(const __m128i *mask) noexcept
			{
				std::size_t result = 0;
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_i16<N>(mask[i], i * 8);
					result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_epi8(vm)));
				}
				return result;
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t find_first_set(const __m128i *mask) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto bits = static_cast<std::uint16_t>(_mm_movemask_epi8(mask[i]));
					if (bits) return std::countr_zero(bits) / 2 + i * 8;
				}
				DPM_UNREACHABLE();
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t find_last_set(const __m128i *mask) noexcept
			{
				for (std::size_t i = M, k; (k = i--) != 0;)
				{
					auto bits = static_cast<std::uint16_t>(_mm_movemask_epi8(mask[i]));
					switch (N - i * 8)
					{
						case 1: bits <<= 2; [[fallthrough]];
						case 2: bits <<= 2; [[fallthrough]];
						case 3: bits <<= 2; [[fallthrough]];
						case 4: bits <<= 2; [[fallthrough]];
						case 5: bits <<= 2; [[fallthrough]];
						case 6: bits <<= 2; [[fallthrough]];
						case 7: bits <<= 2; [[fallthrough]];
						default: break;
					}
					if (bits) return (k * 8 - 1) - std::countl_zero(bits) / 2;
				}
				DPM_UNREACHABLE();
			}

#ifdef DPM_HAS_SSE4_1
			template<std::size_t M>
			static DPM_SAFE_ARRAY void blend(__m128i *out, const __m128i *a, const __m128i *b, const __m128i *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_epi8(a[i], b[i], m[i]);
			}
#endif
		};
	}
}

#endif