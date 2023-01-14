/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "type.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2)

#include "utility.hpp"

namespace dpm
{
	namespace detail
	{
		template<std::size_t N>
		[[nodiscard]] static __m128d x86_maskzero_vector_f64(__m128d v, std::size_t i) noexcept
		{
			if (N - i == 1)
			{
#if defined(DPM_HAS_SSE4_1)
				return _mm_blend_pd(v, _mm_setzero_pd(), 0b10);
#else
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_and_pd(v, _mm_set_pd(0.0, mask));
#endif
			}
			else
				return v;
		}
		template<std::size_t N>
		[[nodiscard]] static __m128d x86_maskone_vector_f64(__m128d v, std::size_t i) noexcept
		{
			if (N - i == 1)
			{
#if defined(DPM_HAS_SSE4_1)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_blend_pd(v, _mm_set1_pd(mask), 0b10);
#else
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_or_pd(v, _mm_set_pd(mask, 0.0));
#endif
			}
			else
				return v;
		}

		template<std::size_t N>
		struct x86_mask_impl<double, __m128d, N>
		{
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_from(const bool *src, __m128d *dst, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					double f0 = 0.0, f1 = 0.0;
					switch (N - i)
					{
						default: f1 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i]));
					}
					dst[i / 2] = _mm_set_pd(f1, f0);
				}
			}
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_to(bool *dst, const __m128d *src, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					switch (const auto bits = _mm_movemask_pd(src[i / 2]); N - i)
					{
						default: dst[i + 1] = bits & 0b10; [[fallthrough]];
						case 1: dst[i] = bits & 0b01;
					}
				}
			}

			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_from(const bool *src, __m128d *dst, const __m128d *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					double f0 = 0.0, f1 = 0.0;
					switch (N - i)
					{
						default: f1 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i]));
					}
					dst[i / 2] = _mm_and_pd(_mm_set_pd(f1, f0), mask[i / 2]);
				}
			}
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_to(bool *dst, const __m128d *src, const __m128d *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
					const auto bits = _mm_movemask_pd(_mm_and_pd(src[i / 2], mask[i / 2]));
					switch (N - i)
					{
						default: dst[i + 1] = bits & 0b10; [[fallthrough]];
						case 1: dst[i] = bits & 0b01;
					}
				}
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void invert(__m128d *dst, const __m128d *src) noexcept
			{
				const auto mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_pd(src[i], mask);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_and(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_and_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_or(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_or_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_xor(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_pd(a[i], b[i]);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_eq(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = std::bit_cast<__m128i>(a[i]);
					const auto vb = std::bit_cast<__m128i>(b[i]);
					out[i] = std::bit_cast<__m128d>(_mm_cmpeq_epi32(va, vb));
				}
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_ne(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				const auto inv_mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = std::bit_cast<__m128i>(a[i]);
					const auto vb = std::bit_cast<__m128i>(b[i]);
					out[i] = _mm_xor_pd(std::bit_cast<__m128d>(_mm_cmpeq_epi32(va, vb)), inv_mask);
				}
			}

			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool all_of(const __m128d *mask) noexcept
			{
#ifdef DPM_HAS_SSE4_1
				if constexpr (M == 1)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[0], 0);
					return _mm_test_all_ones(std::bit_cast<__m128i>(vm));
				}
#endif

				auto result = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_and_pd(result, vm);
				}
				return _mm_movemask_pd(result) == 0b11;
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool any_of(const __m128d *mask) noexcept
			{
				auto result = _mm_setzero_pd();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_pd(result, vm);
				}
#ifdef DPM_HAS_SSE4_1
				const auto vi = std::bit_cast<__m128i>(result);
				return !_mm_testz_si128(vi, vi);
#else
				return _mm_movemask_pd(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool none_of(const __m128d *mask) noexcept
			{
				auto result = _mm_setzero_pd();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_pd(result, vm);
				}
#ifdef DPM_HAS_SSE4_1
				const auto vi = std::bit_cast<__m128i>(result);
				return _mm_testz_si128(vi, vi);
#else
				return !_mm_movemask_pd(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool some_of(const __m128d *mask) noexcept
			{
				auto any_mask = _mm_setzero_pd(), all_mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = mask[i];
					const auto vmz = x86_maskzero_vector_f64<N>(vm, i * 2);
					const auto vmo = x86_maskone_vector_f64<N>(vm, i * 2);

					all_mask = _mm_and_pd(all_mask, vmo);
					any_mask = _mm_or_pd(any_mask, vmz);
				}
#ifdef DPM_HAS_SSE4_1
				const auto any_vi = std::bit_cast<__m128i>(any_mask);
				const auto all_vi = std::bit_cast<__m128i>(all_mask);
				return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
				return _mm_movemask_pd(any_mask) && _mm_movemask_pd(all_mask) != 0b11;
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t popcount(const __m128d *mask) noexcept
			{
				std::size_t result = 0;
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f64<N>(mask[i], i * 2);
					result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_pd(vm)));
				}
				return result;
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t find_first_set(const __m128d *mask) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
					if (bits) return std::countr_zero(bits) + i * 2;
				}
				DPM_UNREACHABLE();
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t find_last_set(const __m128d *mask) noexcept
			{
				for (std::size_t i = M, k; (k = i--) != 0;)
				{
					auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
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
			static DPM_SAFE_ARRAY void blend(__m128d *out, const __m128d *a, const __m128d *b, const __m128d *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_pd(a[i], b[i], m[i]);
			}
#endif
		};

		template<std::size_t N>
		struct x86_simd_impl<double, __m128d, N> : x86_mask_impl<double, __m128d, N>
		{
#ifdef DPM_HAS_SSE4_1
			using x86_mask_impl<double, __m128d, N>::blend;
#endif

			template<typename U>
			static U &data_at(__m128d *data, std::size_t i) noexcept { return reinterpret_cast<U *>(data)[i]; }
			template<typename U>
			static std::add_const_t<U> &data_at(const __m128d *data, std::size_t i) noexcept { return reinterpret_cast<std::add_const_t<U> *>(data)[i]; }

			template<std::size_t M, typename U, typename F>
			static DPM_SAFE_ARRAY void copy_from(const U *src, __m128d *dst, F) noexcept
			{
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							dst[i / 2] = reinterpret_cast<const __m128d *>(src)[i / 2];
						else
							data_at<double>(dst, i) = src[i];
					}
				else if constexpr (signed_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							dst[i / 2] = x86_cvt_i64_f64(reinterpret_cast<const __m128i *>(src)[i / 2]);
						else
							data_at<double>(dst, i) = static_cast<double>(src[i]);
					}
				else if constexpr (unsigned_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							dst[i / 2] = x86_cvt_u64_f64(reinterpret_cast<const __m128i *>(src)[i / 2]);
						else
							data_at<double>(dst, i) = static_cast<double>(src[i]);
					}
				else
					for (std::size_t i = 0; i < N; ++i) data_at<double>(dst, i) = static_cast<double>(src[i]);
			}
			template<std::size_t M, typename U, typename F>
			static DPM_SAFE_ARRAY void copy_to(U *dst, const __m128d *src, F) noexcept
			{
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							reinterpret_cast<__m128d *>(dst)[i / 2] = src[i / 2];
						else
							dst[i] = data_at<double>(src, i);
					}
				else if constexpr (signed_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							reinterpret_cast<__m128i *>(dst)[i / 2] = x86_cvt_f64_i64(src[i / 2]);
						else
							dst[i] = static_cast<U>(data_at<double>(src, i));
					}
				else if constexpr (unsigned_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						if (N - i != i)
							reinterpret_cast<__m128i *>(dst)[i / 2] = x86_cvt_f64_u64(src[i / 2]);
						else
							dst[i] = static_cast<U>(data_at<double>(src, i));
					}
				else
					for (std::size_t i = 0; i < N; ++i) dst[i] = static_cast<U>(data_at<double>(src, i));
			}

			template<std::size_t M, typename U, typename F>
			static DPM_SAFE_ARRAY void copy_from(const U *src, __m128d *dst, const __m128d *mask, F) noexcept
			{
#ifdef DPM_HAS_AVX
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						dst[i / 2] = _mm_maskload_pd(src + i, mi);
					}
				else if constexpr (signed_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						dst[i / 2] = x86_cvt_i64_f64(std::bit_cast<__m128i>(_mm_maskload_pd(src + i, mi)));
					}
				else if constexpr (unsigned_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						dst[i / 2] = x86_cvt_u64_f64(std::bit_cast<__m128i>(_mm_maskload_pd(src + i, mi)));
					}
				else
#endif
				for (std::size_t i = 0; i < N; ++i) if (data_at<std::int64_t>(mask, i)) data_at<double>(dst, i) = static_cast<double>(src[i]);
			}
			template<std::size_t M, typename U, typename F>
			static DPM_SAFE_ARRAY void copy_to(U *dst, const __m128d *src, const __m128d *mask, F) noexcept
			{
#ifdef DPM_HAS_AVX
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						_mm_maskstore_pd(dst + i, mi, src[i / 2]);
					}
				else if constexpr (signed_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						_mm_maskstore_pd(dst + i, mi, std::bit_cast<__m128d>(x86_cvt_f64_i64(src[i / 2])));
					}
				else if constexpr (unsigned_integral_of_size<U, 8> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						_mm_maskstore_epi64(dst + i, mi, std::bit_cast<__m128d>(x86_cvt_f64_u64(src[i / 2])));
					}
				else
#endif
				if constexpr (std::same_as<U, double>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						const auto vi = std::bit_cast<__m128i>(src[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(dst + i));
					}
				else if constexpr (signed_integral_of_size<U, 8>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						const auto vi = x86_cvt_f64_i64(src[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(dst + i));
					}
				else if constexpr (unsigned_integral_of_size<U, 8>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = std::bit_cast<__m128i>(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						const auto vi = x86_cvt_f64_u64(src[i / 2]);
						_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(dst + i));
					}
				else
					for (std::size_t i = 0; i < N; ++i) if (data_at<std::int64_t>(mask, i)) dst[i] = static_cast<U>(data_at<double>(src, i));
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void inc(__m128d *data) noexcept
			{
				const auto one = _mm_set1_pd(1.0);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_add_pd(data[i], one);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void dec(__m128d *data) noexcept
			{
				const auto one = _mm_set1_pd(1.0);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_sub_pd(data[i], one);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void negate(__m128d *dst, const __m128d *src) noexcept
			{
				const auto mask = _mm_set1_pd(-0.0);
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_pd(src[i], mask);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void add(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_add_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void sub(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_sub_pd(a[i], b[i]);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void mul(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_mul_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void div(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_div_pd(a[i], b[i]);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_eq(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpeq_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_le(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmple_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_ge(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpge_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_lt(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmplt_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_gt(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpgt_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_ne(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpneq_pd(a[i], b[i]);
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void min(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void max(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_max_pd(a[i], b[i]);
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void minmax(__m128d *out_min, __m128d *out_max, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					out_min[i] = _mm_min_pd(a[i], b[i]);
					out_max[i] = _mm_max_pd(a[i], b[i]);
				}
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void clamp(__m128d *out, const __m128d *value, const __m128d *min, const __m128d *max) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_pd(_mm_max_pd(value[i], min[i]), max[i]);
			}
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_type<simd_mask<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_size<simd_mask<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
	class simd_mask<double, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		using impl_t = detail::x86_mask_impl<double, __m128d, detail::avec<N, Align>::size>;
		using vector_type = __m128d;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];

	public:
		using value_type = bool;
		using reference = detail::mask_reference<std::int64_t>;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<double, abi_type>;

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
		constexpr DPM_SAFE_ARRAY simd_mask(__m128d native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE mask vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128d)`. */
		constexpr DPM_SAFE_ARRAY simd_mask(const __m128d (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		DPM_SAFE_ARRAY simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff)) : _mm_setzero_pd();
			std::fill_n(m_data, data_size, v);
		}
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_SAFE_ARRAY simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (std::same_as<U, value_type> && (OtherAlign == 0 || OtherAlign >= alignment))
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
			return reference{reinterpret_cast<std::int64_t *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const std::int64_t *>(m_data)[i];
		}

		[[nodiscard]] DPM_SAFE_ARRAY simd_mask operator!() const noexcept
		{
			simd_mask result;
			impl_t::template invert<data_size>(result.m_data, m_data);
			return result;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_xor<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend DPM_SAFE_ARRAY simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_and<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend DPM_SAFE_ARRAY simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_or<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend DPM_SAFE_ARRAY simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_xor<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template cmp_eq<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template cmp_ne<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
		struct native_access<simd_mask<double, avec<N, A>>>
		{
			using mask_t = simd_mask<double, avec<N, A>>;

			[[nodiscard]] static std::span<__m128d, mask_t::data_size> to_native_data(mask_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128d, mask_t::data_size> to_native_data(const mask_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd_mask<double, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd_mask<double, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<double, detail::avec<N, A>> blend(
				const simd_mask<double, detail::avec<N, A>> &a,
				const simd_mask<double, detail::avec<N, A>> &b,
				const simd_mask<double, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd_mask<double, detail::avec<N, A>>>;

			simd_mask<double, detail::avec<N, A>> result;
			detail::x86_mask_impl<double, __m128d, N>::template blend<data_size>(
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
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool all_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template all_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template any_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template none_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template some_of<data_size>(ext::to_native_data(mask).data());
	}

	/** Returns the number of `true` elements of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template popcount<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template find_first_set<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template find_last_set<data_size>(ext::to_native_data(mask).data());
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_type<simd<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
		struct native_data_size<simd<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_m128<double, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_m128<double, N, Align>
	class simd<double, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		using impl_t = detail::x86_simd_impl<double, __m128d, detail::avec<N, Align>::size>;
		using vector_type = __m128d;

		constexpr static auto data_size = ext::native_data_size_v<simd>;
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];

	public:
		using value_type = double;
		using reference = detail::simd_reference<value_type>;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<double, abi_type>;

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
		constexpr DPM_SAFE_ARRAY simd(__m128d native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128d)`. */
		constexpr DPM_SAFE_ARRAY simd(const __m128d (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		DPM_SAFE_ARRAY simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_pd(static_cast<double>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		DPM_SAFE_ARRAY simd(G &&gen) noexcept
		{
			detail::generate_n<data_size>(m_data, [&gen]<std::size_t I>(std::integral_constant<std::size_t, I>)
			{
				double f0 = 0.0f, f1 = 0.0f;
				switch (constexpr auto value_idx = I * 2; size() - value_idx)
				{
					default: f1 = std::invoke(gen, std::integral_constant<std::size_t, value_idx + 1>());
					case 1: f0 = std::invoke(gen, std::integral_constant<std::size_t, value_idx>());
				}
				return _mm_set_pd(f1, f0);
			});
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_SAFE_ARRAY simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (OtherAlign == 0 || OtherAlign >= alignment)
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), vector_aligned);
			else if constexpr (OtherAlign != alignof(value_type))
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), overaligned<OtherAlign>);
			else
				copy_from(reinterpret_cast<const U *>(ext::to_native_data(other).data()), element_aligned);
		}
		/** Initializes the underlying elements from \a mem. */
		template<typename U, typename Flags>
		simd(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { copy_from(mem, Flags{}); }

		/** Copies the underlying elements from \a mem. */
		template<typename U, typename Flags>
		DPM_SAFE_ARRAY void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_from<data_size>(mem, m_data, Flags{}); }
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		DPM_SAFE_ARRAY void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_to<data_size>(mem, m_data, Flags{}); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			DPM_ASSERT(i < size());
			return reference{reinterpret_cast<double *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			DPM_ASSERT(i < size());
			return reinterpret_cast<const double *>(m_data)[i];
		}

		DPM_SAFE_ARRAY simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		DPM_SAFE_ARRAY simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		inline DPM_SAFE_ARRAY simd &operator++() noexcept
		{
			impl_t::template inc<data_size>(m_data);
			return *this;
		}
		inline DPM_SAFE_ARRAY simd &operator--() noexcept
		{
			impl_t::template inc<data_size>(m_data);
			return *this;
		}

		[[nodiscard]] simd DPM_SAFE_ARRAY operator+() const noexcept { return *this; }
		[[nodiscard]] simd DPM_SAFE_ARRAY operator-() const noexcept
		{
			simd result;
			impl_t::template negate<data_size>(result.m_data, m_data);
			return result;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template add<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template sub<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend DPM_SAFE_ARRAY simd &operator+=(simd &a, const simd &b) noexcept
		{
			impl_t::template add<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend DPM_SAFE_ARRAY simd &operator-=(simd &a, const simd &b) noexcept
		{
			impl_t::template sub<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template mul<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY simd operator/(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template div<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend DPM_SAFE_ARRAY simd &operator*=(simd &a, const simd &b) noexcept
		{
			impl_t::template mul<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend DPM_SAFE_ARRAY simd &operator/=(simd &a, const simd &b) noexcept
		{
			impl_t::template div<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_eq<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator<=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_le<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator>=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_ge<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator<(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_lt<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator>(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_gt<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend DPM_SAFE_ARRAY mask_type operator!=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_ne<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_m128<double, N, A>
		struct native_access<simd<double, avec<N, A>>>
		{
			using simd_t = simd<double, avec<N, A>>;

			[[nodiscard]] static std::span<__m128d, simd_t::data_size> to_native_data(simd_t &x) noexcept { return {x.m_data}; }
			[[nodiscard]] static std::span<const __m128d, simd_t::data_size> to_native_data(const simd_t &x) noexcept { return {x.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd<double, detail::avec<N, A>>>::to_native_data(x);
		}
		/** Returns a constant span of the underlying SSE vectors for \a x. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd<double, detail::avec<N, A>> &x) noexcept requires detail::x86_overload_m128<double, N, A>
		{
			return detail::native_access<simd<double, detail::avec<N, A>>>::to_native_data(x);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> blend(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd_mask<double, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_m128<double, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd<double, detail::avec<N, A>>>;

			simd<double, detail::avec<N, A>> result;
			detail::x86_simd_impl<double, __m128d, N>::template blend<data_size>(
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
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> min(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		detail::x86_simd_impl<double, __m128d, N>::template min<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> max(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		detail::x86_simd_impl<double, __m128d, N>::template max<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::pair<simd<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>> minmax(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<double, detail::avec<N, A>>>;

		std::pair<simd<double, detail::avec<N, A>>, simd<double, detail::avec<N, A>>> result;
		detail::x86_simd_impl<double, __m128d, N>::template minmax<data_size>(
				ext::to_native_data(result.first).data(),
				ext::to_native_data(result.second).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}

	/** Clamps elements of \a x between corresponding elements of \a ming and \a max. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> clamp(
			const simd<double, detail::avec<N, A>> &x,
			const simd<double, detail::avec<N, A>> &min,
			const simd<double, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_m128<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		detail::x86_simd_impl<double, __m128d, N>::template clamp<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(x).data(),
				ext::to_native_data(min).data(),
				ext::to_native_data(max).data()
		);
		return result;
	}
#pragma endregion
}

#endif