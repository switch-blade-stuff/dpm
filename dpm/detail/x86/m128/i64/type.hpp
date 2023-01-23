/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../../fwd.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "../f64/utility.hpp"

namespace dpm
{
	namespace detail
	{
		template<std::size_t N>
		[[nodiscard]] static __m128i x86_maskzero_i64(__m128i v, std::size_t i) noexcept
		{
			if (N - i == 1)
			{
#ifdef DPM_HAS_SSE4_1
				return _mm_blend_epi16(v, _mm_setzero_si128(), 0xf0);
#else
				const auto mask = static_cast<std::int64_t>(0xffff'ffff'ffff'ffff);
				return _mm_and_si128(v, _mm_set_epi64x(0, mask));
#endif
			}
			else
				return v;
		}
		template<std::size_t N>
		[[nodiscard]] static __m128i x86_maskone_vector_i64(__m128i v, std::size_t i) noexcept
		{
			if (N - i == 1)
			{
#ifdef DPM_HAS_SSE4_1
				const auto mask = static_cast<std::int64_t>(0xffff'ffff'ffff'ffff);
				return _mm_blend_epi16(v, _mm_set1_epi64x(mask), 0xf0);
#else
				const auto mask = static_cast<std::int64_t>(0xffff'ffff'ffff'ffff);
				return _mm_or_si128(v, _mm_set_epi64x(mask, 0));
#endif
			}
			else
				return v;
		}

		template<integral_of_size<8> I, std::size_t N>
		struct x86_mask_impl<I, __m128i, N>
		{
			template<std::size_t M, typename F>
			static void DPM_FORCEINLINE copy_from(const bool *src, __m128i *dst, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					std::int64_t i0 = 0, i1 = 0;
					switch (N - i)
					{
						default: i1 = extend_bool<std::int64_t>(src[i + 1]); [[fallthrough]];
						case 1: i0 = extend_bool<std::int64_t>(src[i]);
					}
					dst[i / 2] = _mm_set_epi64x(i1, i0);
				}
			}
			template<std::size_t M, typename F>
			static void DPM_FORCEINLINE copy_to(bool *dst, const __m128i *src, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					switch (const auto bits = _mm_movemask_ps(std::bit_cast<__m128>(src[i / 2])); N - i)
					{
						default: dst[i + 1] = bits & 0b1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0011;
					}
				}
			}

			template<std::size_t M, typename F>
			static void DPM_FORCEINLINE copy_from(const bool *src, __m128i *dst, const __m128i *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					std::int64_t i0 = 0, i1 = 0;
					switch (N - i)
					{
						default: i1 = extend_bool<std::int64_t>(src[i + 1]); [[fallthrough]];
						case 1: i0 = extend_bool<std::int64_t>(src[i]);
					}
					dst[i / 2] = _mm_and_si128(_mm_set_epi64x(i1, i0), mask[i / 2]);
				}
			}
			template<std::size_t M, typename F>
			static void DPM_FORCEINLINE copy_to(bool *dst, const __m128i *src, const __m128i *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					const auto bits = _mm_movemask_ps(_mm_and_ps(std::bit_cast<__m128>(src[i / 2]), std::bit_cast<__m128>(mask[i / 2])));
					switch (N - i)
					{
						default: dst[i + 1] = bits & 0b1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0011;
					}
				}
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE invert(__m128i *dst, const __m128i *src) noexcept
			{
				const auto mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_si128(src[i], mask);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE bit_and(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_and_si128(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE bit_or(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_or_si128(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE bit_xor(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_si128(a[i], b[i]);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE cmp_eq(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpeq_epi32(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE cmp_ne(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				const auto inv_mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_si128(_mm_cmpeq_epi32(a[i], b[i]), inv_mask);
			}

			template<std::size_t M>
			[[nodiscard]] static bool all_of(const __m128i *mask) noexcept
			{
#ifdef DPM_HAS_SSE4_1
				if constexpr (M == 1) return _mm_test_all_ones(x86_maskone_vector_i64<N>(mask[0], 0));
#endif
				auto result = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_i64<N>(mask[i], i * 2);
					result = _mm_and_ps(result, std::bit_cast<__m128>(vm));
				}
				return _mm_movemask_ps(result) == 0b1111;
			}
			template<std::size_t M>
			[[nodiscard]] static bool any_of(const __m128i *mask) noexcept
			{
				auto result = _mm_setzero_si128();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_i64<N>(mask[i], i * 2);
					result = _mm_or_si128(result, vm);
				}
#ifdef DPM_HAS_SSE4_1
				return !_mm_testz_si128(result, result);
#else
				return _mm_movemask_epi8(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static bool none_of(const __m128i *mask) noexcept
			{
				auto result = _mm_setzero_si128();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_i64<N>(mask[i], i * 2);
					result = _mm_or_si128(result, vm);
				}
#ifdef DPM_HAS_SSE4_1
				return _mm_testz_si128(result, result);
#else
				return !_mm_movemask_epi8(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static bool some_of(const __m128i *mask) noexcept
			{
#ifdef DPM_HAS_SSE4_1
				auto any_mask = _mm_setzero_si128(), all_mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = mask[i];
					const auto vmz = x86_maskzero_i64<N>(vm, i * 2);
					const auto vmo = x86_maskone_vector_i64<N>(vm, i * 2);

					all_mask = __mm_or_si128(all_mask, vmo);
					any_mask = _mm_or_si128(any_mask, vmz);
				}
				return !_mm_testz_si128(any_mask, any_mask) && !_mm_test_all_ones(all_mask);
#else
				auto any_mask = _mm_setzero_ps(), all_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = mask[i];
					const auto vmz = x86_maskzero_i64<N>(vm, i * 2);
					const auto vmo = x86_maskone_vector_i64<N>(vm, i * 2);

					all_mask = _mm_and_ps(all_mask, std::bit_cast<__m128>(vmo));
					any_mask = _mm_or_ps(any_mask, std::bit_cast<__m128>(vmz));
				}
				return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static std::size_t popcount(const __m128i *mask) noexcept
			{
				std::size_t result = 0;
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_i64<N>(mask[i], i * 2);
					result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_pd(std::bit_cast<__m128d>(vm))));
				}
				return result;
			}
			template<std::size_t M>
			[[nodiscard]] static std::size_t find_first_set(const __m128i *mask) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(std::bit_cast<__m128d>(mask[i])));
					if (bits) return std::countr_zero(bits) + i * 2;
				}
				DPM_UNREACHABLE();
			}
			template<std::size_t M>
			[[nodiscard]] static std::size_t find_last_set(const __m128i *mask) noexcept
			{
				for (std::size_t i = M, k; (k = i--) != 0;)
				{
					auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(std::bit_cast<__m128d>(mask[i])));
					switch (N - i * 2)
					{
						case 1: bits <<= 1; [[fallthrough]];
						default: bits <<= 2;
					}
					if (bits) return (k * 2 - 1) - std::countl_zero(bits);
				}
				DPM_UNREACHABLE();
			}

#ifdef DPM_HAS_SSE4_1
			template<std::size_t M>
			static void DPM_FORCEINLINE blend(__m128i *out, const __m128i *a, const __m128i *b, const __m128i *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_epi8(a[i], b[i], m[i]);
			}
#endif
		};

		template<integral_of_size<8> I, std::size_t N>
		struct x86_simd_impl<I, __m128i, N> : x86_mask_impl<I, __m128i, N>
		{
			using x86_mask_impl<I, __m128i, N>::bit_and;
			using x86_mask_impl<I, __m128i, N>::bit_or;
			using x86_mask_impl<I, __m128i, N>::bit_xor;
			using x86_mask_impl<I, __m128i, N>::invert;

#ifdef DPM_HAS_SSE4_1
			using x86_mask_impl<I, __m128i, N>::blend;
#endif

			template<typename U>
			static U &data_at(__m128i *data, std::size_t i) noexcept { return reinterpret_cast<U *>(data)[i]; }
			template<typename U>
			static std::add_const_t<U> &data_at(const __m128i *data, std::size_t i) noexcept { return reinterpret_cast<std::add_const_t<U> *>(data)[i]; }

			template<std::size_t M, typename U, typename F>
			static void DPM_FORCEINLINE copy_from(const U *src, __m128i *dst, F) noexcept
			{
				if constexpr (std::same_as<U, I> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							dst[i / 2] = reinterpret_cast<const __m128i *>(src)[i / 2];
						else
							data_at<I>(dst, i) = src[i];
					}
				else if constexpr (std::same_as<std::remove_cvref_t<U>, double> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
						{
							if constexpr (std::is_signed_v<I>)
								dst[i / 2] = x86_cvt_f64_i64(reinterpret_cast<const __m128d *>(src)[i / 2]);
							else
								dst[i / 2] = x86_cvt_f64_u64(reinterpret_cast<const __m128d *>(src)[i / 2]);
						}
						else
							data_at<I>(dst, i) = src[i];
					}
				else
					for (std::size_t i = 0; i < N; ++i) data_at<I>(dst, i) = static_cast<I>(src[i]);
			}
			template<std::size_t M, typename U, typename F>
			static void DPM_FORCEINLINE copy_to(U *dst, const __m128i *src, F) noexcept
			{
				if constexpr (std::same_as<U, I> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							reinterpret_cast<__m128i *>(dst)[i / 2] = src[i / 2];
						else
							dst[i] = data_at<I>(src, i);
					}
				else if constexpr (std::same_as<std::remove_cvref_t<U>, double> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
						{
							if constexpr (std::is_signed_v<I>)
								reinterpret_cast<__m128d *>(dst)[i / 2] = x86_cvt_i64_f64(src[i / 2]);
							else
								reinterpret_cast<__m128d *>(dst)[i / 2] = x86_cvt_u64_f64(src[i / 2]);
						}
						else
							dst[i] = data_at<I>(src, i);
					}
				else
					for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<U>(data_at<I>(src, i));
			}

			template<std::size_t M, typename U, typename F>
			static void DPM_FORCEINLINE copy_from(const U *src, __m128i *dst, const __m128i *mask, F) noexcept
			{
#ifdef DPM_HAS_AVX
				if constexpr (std::same_as<U, I> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_i64<N>(mask[i / 2], i));
						dst[i / 2] = std::bit_cast<__m128i>(_mm_maskload_pd(src + i, mi));
					}
				else if constexpr (std::same_as<std::remove_cvref_t<U>, double> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_i64<N>(mask[i / 2], i));
						if constexpr (std::is_signed_v<I>)
							dst[i / 2] = x86_cvt_f64_i64(_mm_maskload_pd(src + i, mi));
						else
							dst[i / 2] = x86_cvt_f64_u64(_mm_maskload_pd(src + i, mi));
					}
				else
#endif
				for (std::size_t i = 0; i < N; ++i) if (data_at<std::int64_t>(mask, i)) data_at<I>(dst, i) = static_cast<I>(src[i]);
			}
			template<std::size_t M, typename U, typename F>
			static void DPM_FORCEINLINE copy_to(U *dst, const __m128i *src, const __m128i *mask, F) noexcept
			{
#ifdef DPM_HAS_AVX
				if constexpr (std::same_as<U, I> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_i64<N>(mask[i / 2], i));
						_mm_maskstore_pd(dst + i, mi, std::bit_cast<__m128d>(src[i / 2]));
					}
				else if constexpr (std::same_as<std::remove_cvref_t<U>, double> && aligned_tag<F, alignof(__m128i)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_i64<N>(mask[i / 2], i));
						if constexpr (std::is_signed_v<I>)
							_mm_maskstore_pd(dst + i, mi, x86_cvt_i64_f64(src[i / 2]));
						else
							_mm_maskstore_pd(dst + i, mi, x86_cvt_u64_f64(src[i / 2]));
					}
				else
#endif
				{
					if constexpr (std::same_as<U, I>)
						for (std::size_t i = 0; i < N; i += 2)
						{
							const auto mi = x86_maskzero_i64<N>(mask[i / 2], i);
							_mm_maskmoveu_si128(src[i / 2], mi, reinterpret_cast<char *>(dst + i));
						}
					else if constexpr (std::same_as<std::remove_cvref_t<U>, double>)
						for (std::size_t i = 0; i < N; i += 2)
						{
							const auto mi = x86_maskzero_i64<N>(mask[i / 2], i);
							if constexpr (std::is_signed_v<I>)
								_mm_maskmoveu_si128(std::bit_cast<__m128i>(x86_cvt_i64_f64(src[i / 2])), mi, reinterpret_cast<char *>(dst + i));
							else
								_mm_maskmoveu_si128(std::bit_cast<__m128i>(x86_cvt_u64_f64(src[i / 2])), mi, reinterpret_cast<char *>(dst + i));
						}
					else
						for (std::size_t i = 0; i < N; ++i) if (data_at<std::int64_t>(mask, i)) dst[i] = static_cast<U>(data_at<I>(src, i));
				}
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE inc(__m128i *data) noexcept
			{
				const auto one = _mm_set1_epi64x(1);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_add_epi64(data[i], one);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE dec(__m128i *data) noexcept
			{
				const auto one = _mm_set1_epi64x(1);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_sub_epi64(data[i], one);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE negate(__m128i *dst, const __m128i *src) noexcept
			{
				const auto zero = _mm_setzero_si128();
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_sub_epi64(zero, src[i]);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE add(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_add_epi64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE sub(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_sub_epi64(a[i], b[i]);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE lshift(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_sll_epi64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE rshift(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_srl_epi64(a[i], b[i]);
			}

#ifdef DPM_HAS_SSE4_1
			template<std::size_t M>
			static void DPM_FORCEINLINE cmp_eq(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpeq_epi64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE cmp_ne(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				const auto inv_mask = _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_si128(_mm_cmpeq_epi64(a[i], b[i]), inv_mask);
			}
#endif

#ifdef DPM_HAS_AVX512VL
			template<std::size_t M>
			static void DPM_FORCEINLINE mul(__m128i *out, const __m128i *a, const __m128i *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_mullo_epi64(a[i], b[i]);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE min(__m128i *out, const __m128i *a, const __m128i *b) noexcept requires std::signed_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_epi64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE max(__m128i *out, const __m128i *a, const __m128i *b) noexcept requires std::signed_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_max_epi64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE minmax(__m128i *out_min, __m128i *out_max, const __m128i *a, const __m128i *b) noexcept requires std::signed_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					out_min[i] = _mm_min_epi64(a[i], b[i]);
					out_max[i] = _mm_max_epi64(a[i], b[i]);
				}
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE clamp(__m128i *out, const __m128i *value, const __m128i *min, const __m128i *max) noexcept requires std::signed_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_epi64(_mm_max_epi64(value[i], min[i]), max[i]);
			}

			template<std::size_t M>
			static void DPM_FORCEINLINE min(__m128i *out, const __m128i *a, const __m128i *b) noexcept requires std::unsigned_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_epu64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE max(__m128i *out, const __m128i *a, const __m128i *b) noexcept requires std::unsigned_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_max_epu64(a[i], b[i]);
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE minmax(__m128i *out_min, __m128i *out_max, const __m128i *a, const __m128i *b) noexcept requires std::unsigned_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					out_min[i] = _mm_min_epu64(a[i], b[i]);
					out_max[i] = _mm_max_epu64(a[i], b[i]);
				}
			}
			template<std::size_t M>
			static void DPM_FORCEINLINE clamp(__m128i *out, const __m128i *value, const __m128i *min, const __m128i *max) noexcept requires std::unsigned_integral<I>
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_epu64(_mm_max_epu64(value[i], min[i]), max[i]);
			}
#endif
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t Align> requires detail::x86_overload_m128<I, N, Align>
		struct native_data_type<simd_mask<I, detail::avec<N, Align>>> { using type = __m128i; };
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t Align> requires detail::x86_overload_m128<I, N, Align>
		struct native_data_size<simd_mask<I, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<I, N, 16>()> {};

		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd_mask<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<I, N, A>;
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd_mask<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<I, N, A>;
	}

	template<detail::integral_of_size<8> I, std::size_t N, std::size_t Align> requires detail::x86_overload_m128<I, N, Align>
	class simd_mask<I, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		using impl_t = detail::x86_mask_impl<I, __m128i, detail::avec<N, Align>::size>;
		using vector_type = __m128i;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];
		using value_alias = detail::mask_alias<std::int64_t>;

	public:
		using value_type = bool;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<I, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the SIMD mask object with a native SSE mask vector.
		 * @note This constructor is available for overload resolution only when the SIMD mask contains a single SSE vector. */
		constexpr simd_mask(__m128i native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128i)`. */
		constexpr simd_mask(const __m128i (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_epi64x(static_cast<std::int64_t>(0xffff'ffff'ffff'ffff)) : _mm_setzero_si128();
			std::fill_n(m_data, data_size, v);
		}
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && alignof(decltype(other)) >= alignment)
				std::copy_n(reinterpret_cast<const vector_type *>(ext::to_native_data(other).data()), data_size, m_data);
			else
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename Flags>
		simd_mask(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename Flags>
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_from<data_size>(mem, m_data, Flags{}); }
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_to<data_size>(mem, m_data, Flags{}); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data)[i];
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data)[i];
		}

		[[nodiscard]] simd_mask operator!() const noexcept
		{
			simd_mask result = {};
			impl_t::template invert<data_size>(result.m_data, m_data);
			return result;
		}

		[[nodiscard]] friend simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template bit_xor<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_and<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_or<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_xor<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		[[nodiscard]] friend simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template cmp_eq<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result = {};
			impl_t::template cmp_ne<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A> requires detail::x86_overload_m128<I, N, A>
		struct native_access<simd_mask<I, avec<N, A>>>
		{
			using mask_t = simd_mask<I, avec<N, A>>;

			[[nodiscard]] static std::span<__m128i, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128i, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<I, N, A>
		{
			return detail::native_access<simd_mask<I, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd_mask<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<I, N, A>
		{
			return detail::native_access<simd_mask<I, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<I, detail::avec<N, A>> blend(
				const simd_mask<I, detail::avec<N, A>> &a,
				const simd_mask<I, detail::avec<N, A>> &b,
				const simd_mask<I, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<I, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd_mask<I, detail::avec<N, A>>>;

			simd_mask<I, detail::avec<N, A>> result = {};
			detail::x86_mask_impl<I, __m128i, N>::template blend<data_size>(
					to_native_data(result).data(),
					to_native_data(a).data(),
					to_native_data(b).data(),
					to_native_data(m).data()
			);
			return result;
		}
#endif
	}

#pragma region "simd_mask reductions"
	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline bool all_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template all_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template any_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template none_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template some_of<data_size>(ext::to_native_data(mask).data());
	}

	/** Returns the number of `true` elements of \a mask. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template popcount<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template find_first_set<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<I, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<I, detail::avec<N, A>>>;
		return detail::x86_mask_impl<I, __m128i, N>::template find_last_set<data_size>(ext::to_native_data(mask).data());
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t Align> requires detail::x86_overload_m128<I, N, Align>
		struct native_data_type<simd<I, detail::avec<N, Align>>> { using type = __m128i; };
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t Align> requires detail::x86_overload_m128<I, N, Align>
		struct native_data_size<simd<I, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<I, N, 16>()> {};

		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<I, N, A>;
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd<I, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<I, N, A>;
	}

	template<detail::integral_of_size<8> I, std::size_t N, std::size_t Align> requires detail::x86_overload_m128<I, N, Align>
	class simd<I, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		using impl_t = detail::x86_simd_impl<I, __m128i, detail::avec<N, Align>::size>;
		using vector_type = __m128i;

		constexpr static auto data_size = ext::native_data_size_v<simd>;
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];
		using value_alias = detail::simd_alias<I>;

	public:
		using value_type = I;
		using reference = value_alias &;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<I, abi_type>;

		/** Returns width of the SIMD type. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd() noexcept = default;
		constexpr simd(const simd &) noexcept = default;
		constexpr simd &operator=(const simd &) noexcept = default;
		constexpr simd(simd &&) noexcept = default;
		constexpr simd &operator=(simd &&) noexcept = default;

		/** Initializes the SIMD vector with a native SSE vector.
		 * @note This constructor is available for overload resolution only when the SIMD vector contains a single SSE vector. */
		constexpr simd(__m128i native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128i)`. */
		constexpr simd(const __m128i (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_epi64x(std::bit_cast<std::int64_t>(static_cast<I>(value)));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		simd(G &&gen) noexcept
		{
			detail::generate_n<data_size>(m_data, [&gen]<std::size_t J>(std::integral_constant<std::size_t, J>)
			{
				I i0 = 0, i1 = 0;
				switch (constexpr auto value_idx = J * 2; size() - value_idx)
				{
					default: i1 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>());
					case 1: i0 = std::invoke(gen, std::integral_constant<std::size_t, value_idx>());
				}
				return _mm_set_epi64x(std::bit_cast<std::int64_t>(i1), std::bit_cast<std::int64_t>(i0));
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (constexpr auto other_alignment = alignof(decltype(other)); other_alignment >= alignment)
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), vector_aligned);
			else if constexpr (other_alignment != alignof(value_type))
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), overaligned<other_alignment>);
			else
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_from<data_size>(mem, m_data, Flags{}); }
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_to<data_size>(mem, m_data, Flags{}); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<value_alias *>(m_data)[i];
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const value_alias *>(m_data)[i];
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
		simd &operator++() noexcept
		{
			impl_t::template inc<data_size>(m_data);
			return *this;
		}
		simd &operator--() noexcept
		{
			impl_t::template inc<data_size>(m_data);
			return *this;
		}

		[[nodiscard]] mask_type operator!() const noexcept
		{
			mask_type result = {};
			for (std::size_t i = 0; i < size(); ++i) result[i] = !static_cast<bool>(operator[](i));
			return result;
		}
		[[nodiscard]] simd operator~() const noexcept
		{
			simd result = {};
			impl_t::template invert<data_size>(result.m_data, m_data);
			return result;
		}
		[[nodiscard]] simd operator+() const noexcept { return *this; }
		[[nodiscard]] simd operator-() const noexcept requires std::is_signed_v<I>
		{
			simd result = {};
			impl_t::template negate<data_size>(result.m_data, m_data);
			return result;
		}

		[[nodiscard]] friend simd operator&(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd operator|(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd operator^(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template bit_xor<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend simd &operator&=(simd &a, const simd &b) noexcept
		{
			impl_t::template bit_and<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend simd &operator|=(simd &a, const simd &b) noexcept
		{
			impl_t::template bit_or<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend simd &operator^=(simd &a, const simd &b) noexcept
		{
			impl_t::template bit_xor<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template add<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template sub<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend simd &operator+=(simd &a, const simd &b) noexcept
		{
			impl_t::template add<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend simd &operator-=(simd &a, const simd &b) noexcept
		{
			impl_t::template sub<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend simd operator<<(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template lshift<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend simd operator>>(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template rshift<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend simd &operator<<=(simd &a, const simd &b) noexcept
		{
			impl_t::template lshift<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend simd &operator>>=(simd &a, const simd &b) noexcept
		{
			impl_t::template rshift<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

#ifdef DPM_HAS_AVX512VL
		[[nodiscard]] friend simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result = {};
			impl_t::template mul<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		friend simd &operator*=(simd &a, const simd &b) noexcept
		{
			impl_t::template mul<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
#endif

#ifdef DPM_HAS_SSE4_1
		[[nodiscard]] friend mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data = {};
			impl_t::template cmp_eq<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data = {};
			impl_t::template cmp_ne<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
#endif

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A> requires detail::x86_overload_m128<I, N, A>
		struct native_access<simd<I, avec<N, A>>>
		{
			using simd_t = simd<I, avec<N, A>>;

			[[nodiscard]] static std::span<__m128i, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128i, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128i, detail::align_data<I, N, 16>()> to_native_data(simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<I, N, A>
		{
			return detail::native_access<simd<I, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128i, detail::align_data<I, N, 16>()> to_native_data(const simd<I, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<I, N, A>
		{
			return detail::native_access<simd<I, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<I, detail::avec<N, A>> blend(
				const simd<I, detail::avec<N, A>> &a,
				const simd<I, detail::avec<N, A>> &b,
				const simd_mask<I, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<I, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd<I, detail::avec<N, A>>>;

			simd<I, detail::avec<N, A>> result = {};
			detail::x86_simd_impl<I, __m128i, N>::template blend<data_size>(
					to_native_data(result).data(),
					to_native_data(a).data(),
					to_native_data(b).data(),
					to_native_data(m).data()
			);
			return result;
		}
#endif
	}

#pragma region "simd algorithms"
#ifdef DPM_HAS_AVX512VL
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<I, detail::avec<N, A>> min(
			const simd<I, detail::avec<N, A>> &a,
			const simd<I, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result = {};
		detail::x86_simd_impl<I, __m128i, N>::template min<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<I, detail::avec<N, A>> max(
			const simd<I, detail::avec<N, A>> &a,
			const simd<I, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		simd<I, detail::avec<N, A>> result = {};
		detail::x86_simd_impl<I, __m128i, N>::template max<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline std::pair<simd<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>> minmax(
			const simd<I, detail::avec<N, A>> &a,
			const simd<I, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		std::pair<simd<I, detail::avec<N, A>>, simd<I, detail::avec<N, A>>> result = {};
		detail::x86_simd_impl<I, __m128i, N>::template minmax<data_size>(
				ext::to_native_data(result.first).data(),
				ext::to_native_data(result.second).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}

	/** Clamps elements of \a x between corresponding elements of \a ming and \a max. */
	template<detail::integral_of_size<8> I, std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<I, detail::avec<N, A>> clamp(
			const simd<I, detail::avec<N, A>> &x,
			const simd<I, detail::avec<N, A>> &min,
			const simd<I, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_m128<I, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<I, detail::avec<N, A>>>;

		simd<I, detail::avec<N, A>> result = {};
		detail::x86_simd_impl<I, __m128i, N>::template clamp<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(x).data(),
				ext::to_native_data(min).data(),
				ext::to_native_data(max).data()
		);
		return result;
	}
#endif
#pragma endregion
}

#endif