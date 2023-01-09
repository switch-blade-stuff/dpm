/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../fwd.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE) || defined(DPM_DYNAMIC_DISPATCH))

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
#elif defined(DPM_HAS_SSE2)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_and_pd(v, _mm_set_pd(0.0, mask));
#else
				const auto mask = std::bit_cast<float>(0xffff'ffff);
				const auto maskv = _mm_set_ps(0.0f, 0.0f, mask, mask);
				return std::bit_cast<__m128d>(_mm_and_ps(std::bit_cast<__m128>(v), maskv));
#endif
			}
			else
				return v;
		}
		template<std::size_t N>
		[[nodiscard]] static __m128d x86_maskone_vector_f32(__m128d v, std::size_t i) noexcept
		{
			if (N - i == 1)
			{
#if defined(DPM_HAS_SSE4_1)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_blend_pd(v, _mm_set1_pd(mask), 0b10);
#elif defined(DPM_HAS_SSE2)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_or_pd(v, _mm_set_pd(mask, 0.0));
#else
				const auto mask = std::bit_cast<float>(0xffff'ffff);
				const auto maskv = _mm_set_ps(mask, mask, 0.0f, 0.0f);
				return std::bit_cast<__m128d>(_mm_or_ps(std::bit_cast<__m128>(v), maskv));
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
#ifdef DPM_HAS_SSE2
					double f0 = 0.0, f1 = 0.0;
					switch (N - i)
					{
						default: f1 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i]));
					}
					dst[i / 2] = _mm_set_pd(f1, f0);
#else
					float f0 = 0.0f, f1 = 0.0f;
					switch (N - i)
					{
						default: f1 = std::bit_cast<float>(extend_bool<std::int32_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<float>(extend_bool<std::int32_t>(src[i]));
					}
					dst[i / 2] = std::bit_cast<__m128d>(_mm_set_ps(f1, f1, f0, f0));
#endif
				}
			}
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_to(bool *dst, const __m128d *src, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef DPM_HAS_SSE2
					switch (const auto bits = _mm_movemask_pd(src[i / 2]); N - i)
					{
						default: dst[i + 1] = bits & 0b10; [[fallthrough]];
						case 1: dst[i] = bits & 0b01;
					}
#else
					switch (const auto bits = _mm_movemask_ps(std::bit_cast<__m128>(src[i / 2])); N - i)
					{
						default: dst[i + 1] = bits & 0b1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0011;
					}
#endif
				}
			}

			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_from(const bool *src, __m128d *dst, const __m128d *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef DPM_HAS_SSE2
					double f0 = 0.0, f1 = 0.0;
					switch (N - i)
					{
						default: f1 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<double>(extend_bool<std::int64_t>(src[i]));
					}
					dst[i / 2] = _mm_and_pd(_mm_set_pd(f1, f0), mask[i / 2]);
#else
					float f0 = 0.0f, f1 = 0.0f;
					switch (N - i)
					{
						default: f1 = std::bit_cast<float>(extend_bool<std::int32_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<float>(extend_bool<std::int32_t>(src[i]));
					}
					dst[i / 2] = std::bit_cast<__m128d>(_mm_and_ps(_mm_set_ps(f1, f1, f0, f0), std::bit_cast<__m128>(mask[i / 2])));
#endif
				}
			}
			template<std::size_t M, typename F>
			static DPM_SAFE_ARRAY void copy_to(bool *dst, const __m128d *src, const __m128d *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef DPM_HAS_SSE2
					const auto bits = _mm_movemask_ps(_mm_and_ps(src[i / 2], mask[i / 2]));
					switch (N - i)
					{
						default: dst[i + 1] = bits & 0b10; [[fallthrough]];
						case 1: dst[i] = bits & 0b01;
					}
#else
					const auto bits = _mm_movemask_ps(_mm_and_ps(std::bit_cast<__m128>(src[i / 2]), std::bit_cast<__m128>(mask[i / 2])));
					switch (N - i)
					{
						default: dst[i + 1] = bits & 0b1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0011;
					}
#endif
				}
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void invert(__m128d *dst, const __m128d *src) noexcept
			{
#ifdef DPM_HAS_SSE2
				const auto mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_pd(src[i], mask);
#else
				const auto mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = std::bit_cast<__m128d>(_mm_xor_ps(std::bit_cast<__m128>(src[i]), mask));
#endif
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_and(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef DPM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_and_pd(a[i], b[i]);
#else
				for (std::size_t i = 0; i < M; ++i) dst[i] = std::bit_cast<__m128d>(_mm_and_ps(std::bit_cast<__m128>(a[i]), std::bit_cast<__m128>(b[i])));
#endif
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_or(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef DPM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_or_pd(a[i], b[i]);
#else
				for (std::size_t i = 0; i < M; ++i) dst[i] = std::bit_cast<__m128d>(_mm_or_ps(std::bit_cast<__m128>(a[i]), std::bit_cast<__m128>(b[i])));
#endif
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void bit_xor(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef DPM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_pd(a[i], b[i]);
#else
				for (std::size_t i = 0; i < M; ++i) dst[i] = std::bit_cast<__m128d>(_mm_xor_ps(std::bit_cast<__m128>(a[i]), std::bit_cast<__m128>(b[i])));
#endif
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_eq(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef DPM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_castpd_si128(a[i]);
					const auto vb = _mm_castpd_si128(b[i]);
					out[i] = _mm_castsi128_pd(_mm_cmpeq_epi32(va, vb));
				}
#else
				const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_and_ps(std::bit_cast<__m128>(a[i]), nan_mask);
					const auto vb = _mm_and_ps(std::bit_cast<__m128>(b[i]), nan_mask);
					fo[i] = std::bit_cast<__m128d>(_mm_cmpeq_ps(va, vb));
				}
#endif
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void cmp_ne(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef DPM_HAS_SSE2
				const auto inv_mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_castpd_si128(a[i]);
					const auto vb = _mm_castpd_si128(b[i]);
					out[i] = _mm_xor_pd(_mm_castsi128_pd(_mm_cmpeq_epi32(va, vb)), inv_mask);
				}
#else
				const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_and_ps(std::bit_cast<__m128>(a[i]), nan_mask);
					const auto vb = _mm_and_ps(std::bit_cast<__m128>(b[i]), nan_mask);
					fo[i] = std::bit_cast<__m128d>(_mm_cmpneq_ps(va, vb));
				}
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool all_of(const __m128d *mask) noexcept
			{
#ifdef DPM_HAS_SSE4_1
				if constexpr (M == 1)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[0], 0);
					return _mm_test_all_ones(_mm_castpd_si128(vm));
				}
#endif

#ifdef DPM_HAS_SSE2
				auto result = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_and_pd(result, vm);
				}
				return _mm_movemask_pd(result) == 0b11;
#else
				auto result = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_and_ps(result, std::bit_cast<__m128>(vm));
				}
				return _mm_movemask_ps(result) == 0b1111;
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool any_of(const __m128d *mask) noexcept
			{
#ifdef DPM_HAS_SSE2
				auto result = _mm_setzero_pd();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_pd(result, vm);
				}
#ifdef DPM_HAS_SSE4_1
				const auto vi = _mm_castpd_si128(result);
				return !_mm_testz_si128(vi, vi);
#else
				return _mm_movemask_pd(result);
#endif
#else
				auto result = _mm_setzero_ps();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_ps(result, std::bit_cast<__m128>(vm));
				}
				return _mm_movemask_ps(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool none_of(const __m128d *mask) noexcept
			{
#ifdef DPM_HAS_SSE2
				auto result = _mm_setzero_pd();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_pd(result, vm);
				}
#ifdef DPM_HAS_SSE4_1
				const auto vi = _mm_castpd_si128(result);
				return _mm_testz_si128(vi, vi);
#else
				return !_mm_movemask_pd(result);
#endif
#else
				auto result = _mm_setzero_ps();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_ps(result, std::bit_cast<__m128>(vm));
				}
				return !_mm_movemask_ps(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY bool some_of(const __m128d *mask) noexcept
			{
#ifdef DPM_HAS_SSE2
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
				const auto any_vi = _mm_castpd_si128(any_mask);
				const auto all_vi = _mm_castpd_si128(all_mask);
				return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
				return _mm_movemask_pd(any_mask) && _mm_movemask_pd(all_mask) != 0b11;
#endif
#else
				auto any_mask = _mm_setzero_ps(), all_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = mask[i];
					const auto vmz = x86_maskzero_vector_f64<N>(vm, i * 2);
					const auto vmo = x86_maskone_vector_f64<N>(vm, i * 2);

					all_mask = _mm_and_ps(all_mask, std::bit_cast<__m128>(vmo));
					any_mask = _mm_or_ps(any_mask, std::bit_cast<__m128>(vmz));
				}
				return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t popcount(const __m128d *mask) noexcept
			{
				std::size_t result = 0;
				for (std::size_t i = 0; i < M; ++i)
				{
#ifdef DPM_HAS_SSE2
					const auto vm = x86_maskzero_vector_f64<N>(mask[i], i * 2);
					result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_pd(vm)));
#else
					const auto vm = std::bit_cast<__m128>(x86_maskzero_vector_f64<N>(mask[i], i * 2));
					result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_ps(vm) & 0b0101));
#endif
				}
				return result;
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t find_first_set(const __m128d *mask) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
#ifdef DPM_HAS_SSE2
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
					if (bits) return std::countr_zero(bits) + i * 2;
#else
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
					if (bits) return std::countr_zero((bits & 0b0110) >> 1) + i * 2;
#endif
				}
				DPM_UNREACHABLE();
			}
			template<std::size_t M>
			[[nodiscard]] static DPM_SAFE_ARRAY std::size_t find_last_set(const __m128d *mask) noexcept
			{
				for (std::size_t i = M, k; (k = i--) != 0;)
				{
#ifdef DPM_HAS_SSE2
					auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
					switch (N - i * 2)
					{
						case 1: bits <<= 1; [[fallthrough]];
						default: bits <<= 2;
					}
#else
					auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(mask[i]));
					switch (N - i * 2)
					{
						case 1: bits <<= 2; [[fallthrough]];
						default: bits = (bits & 0b0110) << 5;
					}
#endif
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
		struct x86_simd_impl<double, __m128d, N>
		{
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
						const auto mi = _mm_castpd_si128(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						dst[i / 2] = _mm_maskload_pd(src + i, mi);
					}
				else
#endif
					for (std::size_t i = 0; i < N; ++i) if (data_at<std::uint64_t>(mask, i)) data_at<double>(dst, i) = static_cast<double>(src[i]);
			}
			template<std::size_t M, typename U, typename F>
			static DPM_SAFE_ARRAY void copy_to(U *dst, const __m128d *src, const __m128d *mask, F) noexcept
			{
#ifdef DPM_HAS_AVX
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 2)
					{
						const auto mi = _mm_castpd_si128(x86_maskzero_vector_f64<N>(mask[i / 2], i));
						_mm_maskstore_pd(dst + i, mi, src[i / 2]);
					}
				else
#endif
				{
#ifdef DPM_HAS_SSE2
					if constexpr (std::same_as<U, double>)
						for (std::size_t i = 0; i < N; i += 2)
						{
							const auto mi = _mm_castpd_si128(x86_maskzero_vector_f64<N>(mask[i / 2], i));
							const auto vi = _mm_castpd_si128(src[i / 2]);
							_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(dst + i));
						}
					else
#endif
						for (std::size_t i = 0; i < N; ++i) if (data_at<std::uint64_t>(mask, i)) dst[i] = static_cast<U>(data_at<double>(src, i));
				}
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void inc(__m128d *data) noexcept
			{
#ifdef DPM_HAS_SSE2
				const auto one = _mm_set1_pd(1.0);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_add_pd(data[i], one);
#else
				for (std::size_t i = 0; i < N; ++i) ++data_at<double>(data, i);
#endif
			}
			template<std::size_t M>
			static DPM_SAFE_ARRAY void dec(__m128d *data) noexcept
			{
#ifdef DPM_HAS_SSE2
				const auto one = _mm_set1_pd(1.0);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_sub_pd(data[i], one);
#else
				for (std::size_t i = 0; i < N; ++i) --data_at<double>(data, i);
#endif
			}

			template<std::size_t M>
			static DPM_SAFE_ARRAY void negate(__m128d *dst, const __m128d *src) noexcept
			{
#ifdef DPM_HAS_SSE2
				const auto mask = _mm_set1_pd(std::bit_cast<double>(0x8000'0000'0000'0000));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_pd(src[i], mask);
#else
				for (std::size_t i = 0; i < N; ++i) data_at<double>(dst, i) = -data_at<double>(src, i);
#endif
			}

#ifdef DPM_HAS_SSE2
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

#ifdef DPM_HAS_SSE4_1
			template<std::size_t M>
			static DPM_SAFE_ARRAY void blend(__m128d *out, const __m128d *a, const __m128d *b, const __m128d *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_pd(a[i], b[i], m[i]);
			}
#endif

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
#endif
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<double, N, Align>
		struct native_data_type<simd_mask<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<double, N, Align>
		struct native_data_size<simd_mask<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_sse<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_sse<double, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_sse<double, N, Align>
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
		template<std::size_t N, std::size_t A> requires detail::x86_overload_sse<double, N, A>
		struct native_access<simd_mask<double, avec<N, A>>>
		{
			using mask_t = simd_mask<double, avec<N, A>>;

			[[nodiscard]] static std::span<__m128d, mask_t::data_size> to_native_data(mask_t &value) noexcept { return {value.m_data}; }
			[[nodiscard]] static std::span<const __m128d, mask_t::data_size> to_native_data(const mask_t &value) noexcept { return {value.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<double, N, A>
		{
			return detail::native_access<simd_mask<double, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a constant span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<double, N, A>
		{
			return detail::native_access<simd_mask<double, detail::avec<N, A>>>::to_native_data(value);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<double, detail::avec<N, A>> blend(
				const simd_mask<double, detail::avec<N, A>> &a,
				const simd_mask<double, detail::avec<N, A>> &b,
				const simd_mask<double, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_sse<double, N, A>
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
	[[nodiscard]] inline bool all_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template all_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template any_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template none_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template some_of<data_size>(ext::to_native_data(mask).data());
	}

	/** Returns the number of `true` elements of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template popcount<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template find_first_set<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<double, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<double, detail::avec<N, A>>>;
		return detail::x86_mask_impl<double, __m128d, N>::template find_last_set<data_size>(ext::to_native_data(mask).data());
	}
#pragma endregion

	DPM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<double, N, Align>
		struct native_data_type<simd<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<double, N, Align>
		struct native_data_size<simd<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_sse<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd<double, detail::avec<N, A>> &) noexcept requires detail::x86_overload_sse<double, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_sse<double, N, Align>
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
#ifdef DPM_HAS_SSE2
			const auto vec = _mm_set1_pd(static_cast<double>(value));
			std::fill_n(m_data, data_size, vec);
#else
			std::fill_n(reinterpret_cast<double *>(m_data), size(), static_cast<double>(value));
#endif
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		DPM_SAFE_ARRAY simd(G &&gen) noexcept
		{
#ifdef DPM_HAS_SSE2
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
#else
			detail::generate_n<size()>(reinterpret_cast<double *>(m_data), std::forward<G>(gen));
#endif
		}

		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		DPM_SAFE_ARRAY simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
		{
			if constexpr (!std::same_as<U, value_type>)
				for (std::size_t i = 0; i < size(); ++i) operator[](i) = other[i];
			else if constexpr (OtherAlign == 0 || OtherAlign >= alignment)
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

#ifdef DPM_HAS_SSE2
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
#endif

	private:
		alignas(alignment) data_type m_data;
	};

	namespace detail
	{
		template<std::size_t N, std::size_t A> requires detail::x86_overload_sse<double, N, A>
		struct native_access<simd<double, avec<N, A>>>
		{
			using simd_t = simd<double, avec<N, A>>;

			[[nodiscard]] static std::span<__m128d, simd_t::data_size> to_native_data(simd_t &value) noexcept { return {value.m_data}; }
			[[nodiscard]] static std::span<const __m128d, simd_t::data_size> to_native_data(const simd_t &value) noexcept { return {value.m_data}; }
		};
	}

	DPM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<double, N, A>
		{
			return detail::native_access<simd<double, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a constant span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<double, N, A>
		{
			return detail::native_access<simd<double, detail::avec<N, A>>>::to_native_data(value);
		}

#ifdef DPM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<double, detail::avec<N, A>> blend(
				const simd<double, detail::avec<N, A>> &a,
				const simd<double, detail::avec<N, A>> &b,
				const simd_mask<double, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_sse<double, N, A>
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
#ifdef DPM_HAS_SSE2
	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> min(
			const simd<double, detail::avec<N, A>> &a,
			const simd<double, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_sse<double, N, A>
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
	noexcept requires detail::x86_overload_sse<double, N, A>
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
	noexcept requires detail::x86_overload_sse<double, N, A>
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

	/** Clamps elements \f \a value between corresponding elements of \a ming and \a max. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<double, detail::avec<N, A>> clamp(
			const simd<double, detail::avec<N, A>> &value,
			const simd<double, detail::avec<N, A>> &min,
			const simd<double, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_sse<double, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<double, detail::avec<N, A>>>;

		simd<double, detail::avec<N, A>> result;
		detail::x86_simd_impl<double, __m128d, N>::template clamp<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(value).data(),
				ext::to_native_data(min).data(),
				ext::to_native_data(max).data()
		);
		return result;
	}
#endif
#pragma endregion
}

#endif