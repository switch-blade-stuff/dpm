/*
 * Created by switchblade on 2023-01-06.
 */

#pragma once

#include "../fwd.hpp"

#if defined(SVM_ARCH_X86) && (defined(SVM_HAS_SSE) || defined(SVM_DYNAMIC_DISPATCH))

namespace svm
{
	namespace detail
	{
		template<std::size_t N>
		static __m128d x86_maskzero_vector_f64(__m128d v, std::size_t i) noexcept
		{
			if constexpr (N - i == 1)
			{
#if defined(SVM_HAS_SSE4_1)
				return _mm_blend_pd(v, _mm_setzero_pd(), 0b10);
#elif defined(SVM_HAS_SSE2)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_and_pd(v, _mm_set_pd(0.0f, mask));
#else
				const auto mask = std::bit_cast<float>(0xffff'ffff);
				return _mm_and_ps(v, _mm_set_ps(0.0f, 0.0f, mask, mask));
#endif
			}
			else
				return v;
		}
		template<std::size_t N>
		static __m128d x86_maskone_vector_f32(__m128d v, std::size_t i) noexcept
		{
			if constexpr (N - i == 1)
			{
#if defined(SVM_HAS_SSE4_1)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_blend_pd(v, _mm_set1_pd(mask), 0b10);
#elif defined(SVM_HAS_SSE2)
				const auto mask = std::bit_cast<double>(0xffff'ffff'ffff'ffff);
				return _mm_or_pd(v, _mm_set_pd(mask, 0.0f));
#else
				const auto mask = std::bit_cast<float>(0xffff'ffff);
				return _mm_or_ps(v, _mm_set_ps(mask, mask, 0.0f, 0.0f));
#endif
			}
			else
				return v;
		}

		template<std::size_t N>
		struct x86_impl<bool, __m128d, N>
		{
			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_from(const bool *src, __m128d *dst, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef SVM_HAS_SSE2
					double f0 = 0.0f, f1 = 0.0f;
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
					reinterpret_cast<__m128 *>(dst)[i / 2] = _mm_set_pf(f1, f1, f0, f0);
#endif
				}
			}
			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_to(bool *dst, const __m128d *src, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef SVM_HAS_SSE2
					switch (const auto bits = _mm_movemask_pd(src[i / 2]); N - i)
					{
						default: dst[i + 1] = bits & 0b10; [[fallthrough]];
						case 1: dst[i] = bits & 0b01;
					}
#else
					switch (const auto bits = _mm_movemask_ps(reinterpret_cast<const __m128 *>(src)[i / 2]); N - i)
					{
						default: dst[i + 1] = bits & 0b1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0011;
					}
#endif
				}
			}

			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_from(const bool *src, __m128d *dst, const __m128d *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef SVM_HAS_SSE2
					double f0 = 0.0f, f1 = 0.0f;
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
					const auto fm = reinterpret_cast<const __m128 *>(mask);
					const auto fd = reinterpret_cast<__m128 *>(dst);
					fd[i / 2] = _mm_and_ps(_mm_set_ps(f1, f1, f0, f0), fm[i / 2]);
#endif
				}
			}
			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_to(bool *dst, const __m128d *src, const __m128d *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 2)
				{
#ifdef SVM_HAS_SSE2
					const auto bits = _mm_movemask_ps(_mm_and_ps(src[i / 2], mask[i / 2]));
					switch (N - i)
					{
						default: dst[i + 1] = bits & 0b10; [[fallthrough]];
						case 1: dst[i] = bits & 0b01;
					}
#else
					const auto fm = reinterpret_cast<const __m128 *>(mask);
					const auto fs = reinterpret_cast<const __m128 *>(src);
					const auto bits = _mm_movemask_ps(_mm_and_ps(fs[i / 2], fm[i / 2]));
					switch (N - i)
					{
						default: dst[i + 1] = bits & 0b1100; [[fallthrough]];
						case 1: dst[i] = bits & 0b0011;
					}
#endif
				}
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void invert(__m128d *dst, const __m128d *src) noexcept
			{
#ifdef SVM_HAS_SSE2
				const auto mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_pd(src[i], mask);
#else
				const auto mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto fs = reinterpret_cast<const __m128 *>(src);
					const auto fd = reinterpret_cast<__m128 *>(dst);
					fd[i] = _mm_xor_ps(fs[i], mask);
				}
#endif
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void bit_and(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef SVM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_and_pd(a[i], b[i]);
#else
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto fs = reinterpret_cast<const __m128 *>(src);
					const auto fd = reinterpret_cast<__m128 *>(dst);
					fd[i] = _mm_and_ps(fs[i], mask);
				}
#endif
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void bit_or(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef SVM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_or_pd(a[i], b[i]);
#else
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto fs = reinterpret_cast<const __m128 *>(src);
					const auto fd = reinterpret_cast<__m128 *>(dst);
					fd[i] = _mm_or_ps(fs[i], mask);
				}
#endif
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void bit_xor(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef SVM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_pd(a[i], b[i]);
#else
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto fs = reinterpret_cast<const __m128 *>(src);
					const auto fd = reinterpret_cast<__m128 *>(dst);
					fd[i] = _mm_xor_ps(fs[i], mask);
				}
#endif
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_eq(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef SVM_HAS_SSE2
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
					const auto fa = reinterpret_cast<const __m128 *>(a);
					const auto fb = reinterpret_cast<const __m128 *>(b);
					const auto fo = reinterpret_cast<__m128 *>(out);

					const auto va = _mm_and_ps(fa[i], nan_mask);
					const auto vb = _mm_and_ps(fb[i], nan_mask);
					fo[i] = _mm_cmpeq_ps(va, vb);
				}
#endif
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_ne(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
#ifdef SVM_HAS_SSE2
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
					const auto fa = reinterpret_cast<const __m128 *>(a);
					const auto fb = reinterpret_cast<const __m128 *>(b);
					const auto fo = reinterpret_cast<__m128 *>(out);

					const auto va = _mm_and_ps(fa[i], nan_mask);
					const auto vb = _mm_and_ps(fb[i], nan_mask);
					fo[i] = _mm_cmpneq_ps(va, vb);
				}
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY bool all_of(const __m128d *mask) noexcept
			{
#ifdef SVM_HAS_SSE4_1
				if constexpr (M == 1)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[0], 0);
					return _mm_test_all_ones(_mm_castpd_si128(vm));
				}
#endif

#ifdef SVM_HAS_SSE2
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
			[[nodiscard]] static SVM_SAFE_ARRAY bool any_of(const __m128d *mask) noexcept
			{
#ifdef SVM_HAS_SSE2
				auto result = _mm_setzero_pd();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_pd(result, vm);
				}
#ifdef SVM_HAS_SSE4_1
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
			[[nodiscard]] static SVM_SAFE_ARRAY bool none_of(const __m128d *mask) noexcept
			{
#ifdef SVM_HAS_SSE2
				auto result = _mm_setzero_pd();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f64<N>(mask[i], i * 2);
					result = _mm_or_pd(result, vm);
				}
#ifdef SVM_HAS_SSE4_1
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
			[[nodiscard]] static SVM_SAFE_ARRAY bool some_of(const __m128d *mask) noexcept
			{
#ifdef SVM_HAS_SSE2
				auto any_mask = _mm_setzero_pd(), all_mask = _mm_set1_pd(std::bit_cast<double>(0xffff'ffff'ffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = mask[i];
					const auto vmz = x86_maskzero_vector_f64<N>(vm, i * 2);
					const auto vmo = x86_maskone_vector_f64<N>(vm, i * 2);

					all_mask = _mm_and_pd(all_mask, vmo);
					any_mask = _mm_or_pd(any_mask, vmz);
				}
#ifdef SVM_HAS_SSE4_1
				const auto any_vi = _mm_castpd_si128(any_mask);
				const auto all_vi = _mm_castpd_si128(all_mask);
				return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
				return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
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
			[[nodiscard]] static SVM_SAFE_ARRAY std::size_t popcount(const __m128d *mask) noexcept
			{
				std::size_t result = 0;
				for (std::size_t i = 0; i < M; ++i)
				{
#ifdef SVM_HAS_SSE2
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
			[[nodiscard]] static SVM_SAFE_ARRAY std::size_t find_first_set(const __m128d *mask) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
#ifdef SVM_HAS_SSE2
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
					if (bits) return std::countr_zero(bits) + i * 2;
#else
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_pd(mask[i]));
					if (bits) return std::countr_zero((bits & 0b0110) >> 1) + i * 2;
#endif
				}
				SVM_UNREACHABLE();
			}
			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY std::size_t find_last_set(const __m128d *mask) noexcept
			{
				for (std::size_t i = M, k; (k = i--) != 0;)
				{
#ifdef SVM_HAS_SSE2
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
				SVM_UNREACHABLE();
			}

#ifdef SVM_HAS_SSE4_1
			template<std::size_t M>
			static SVM_SAFE_ARRAY void blend(__m128d *out, const __m128d *a, const __m128d *b, const __m128d *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_pd(a[i], b[i], m[i]);
			}
#endif
		};

#ifdef SVM_HAS_SSE2
		template<std::size_t N>
		struct x86_impl<double, __m128d, N>
		{
			template<typename U>
			static U &data_at(__m128d *data, std::size_t i) noexcept { return reinterpret_cast<U *>(data)[i]; }
			template<typename U>
			static std::add_const_t<U> &data_at(const __m128d *data, std::size_t i) noexcept { return reinterpret_cast<std::add_const_t<U> *>(data)[i]; }

			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_from(const U *src, __m128d *dst, F) noexcept
			{
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: dst[i / 4] = reinterpret_cast<const __m128d *>(src)[i / 4];
								break;
							case 3: data_at<double>(dst, i + 2) = src[i + 2]; [[fallthrough]];
							case 2: data_at<double>(dst, i + 1) = src[i + 1]; [[fallthrough]];
							case 1: data_at<double>(dst, i) = src[i];
						}
				}
#ifdef SVM_HAS_SSE2
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128d)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: dst[i / 4] = _mm_cvtepi32_ps(reinterpret_cast<__m128di *>(src)[i / 4]);
								break;
							case 3: data_at<double>(dst, i + 2) = static_cast<double>(src[i + 2]); [[fallthrough]];
							case 2: data_at<double>(dst, i + 1) = static_cast<double>(src[i + 1]); [[fallthrough]];
							case 1: data_at<double>(dst, i) = static_cast<double>(src[i]);
						}
				}
#endif
				else
					std::copy_n(src, N, reinterpret_cast<double *>(dst));
			}
			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_to(U *dst, const __m128d *src, F) noexcept
			{
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: reinterpret_cast<__m128d *>(dst)[i / 4] = src[i / 4];
								break;
							case 3: dst[i + 2] = data_at<double>(src, i + 2); [[fallthrough]];
							case 2: dst[i + 1] = data_at<double>(src, i + 1); [[fallthrough]];
							case 1: dst[i] = data_at<double>(src, i);
						}
				}
#ifdef SVM_HAS_SSE2
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128d)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: reinterpret_cast<__m128di *>(dst)[i / 4] = _mm_cvtps_epi32(src[i / 4]);
								break;
							case 3: dst[i + 2] = static_cast<std::int32_t>(data_at<double>(src, i + 2)); [[fallthrough]];
							case 2: dst[i + 1] = static_cast<std::int32_t>(data_at<double>(src, i + 1)); [[fallthrough]];
							case 1: dst[i] = static_cast<std::int32_t>(data_at<double>(src, i));
						}
				}
#endif
				else
					std::copy_n(reinterpret_cast<const double *>(src), N, dst);
			}

			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_from(const U *src, __m128d *dst, const __m128d *mask, F) noexcept
			{
#ifdef SVM_HAS_AVX
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						dst[i] = _mm_maskload_ps(src + i, mi);
					}
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						dst[i] = _mm_cvtepi32_ps(_mm_maskload_epi32(src + i, mi));
					}
				else
#endif
					for (std::size_t i = 0; i < N; ++i) if (data_at<std::uint32_t>(mask, i)) data_at<double>(dst, i) = src[i];
			}
			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_to(U *dst, const __m128d *src, const __m128d *mask, F) noexcept
			{
#ifdef SVM_HAS_AVX
				if constexpr (std::same_as<U, double> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						_mm_maskstore_ps(dst + i, mi, src[i / 4]);
					}
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128d)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						_mm_maskstore_epi32(dst + i, mi, _mm_cvtps_epi32(src[i / 4]));
					}
				else
#endif
				{
#ifdef SVM_HAS_SSE2
					if constexpr (std::same_as<U, double>)
						for (std::size_t i = 0; i < N; i += 4)
						{
							const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
							const auto vi = _mm_castps_si128(src[i / 4]);
							_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(dst + i));
						}
					else
#endif
						for (std::size_t i = 0; i < N; ++i) if (data_at<std::uint32_t>(mask, i)) dst[i] = data_at<double>(src, i);
				}
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void inc(__m128d *data) noexcept
			{
				const auto one = _mm_set1_ps(1.0f);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_add_ps(data[i], one);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void dec(__m128d *data) noexcept
			{
				const auto one = _mm_set1_ps(1.0f);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_sub_ps(data[i], one);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void negate(__m128d *dst, const __m128d *src) noexcept
			{
				const auto mask = _mm_set1_ps(std::bit_cast<double>(0x8000'0000));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_ps(src[i], mask);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void add(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_add_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void sub(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_sub_ps(a[i], b[i]);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void mul(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_mul_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void div(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_div_ps(a[i], b[i]);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_eq(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpeq_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_le(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmple_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_ge(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpge_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_lt(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmplt_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_gt(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpgt_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_ne(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpneq_ps(a[i], b[i]);
			}

#ifdef SVM_HAS_SSE4_1
			template<std::size_t M>
			static SVM_SAFE_ARRAY void blend(__m128d *out, const __m128d *a, const __m128d *b, const __m128d *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_ps(a[i], b[i], m[i]);
			}
#endif

			template<std::size_t M>
			static SVM_SAFE_ARRAY void min(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void max(__m128d *out, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_max_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void minmax(__m128d *out_min, __m128d *out_max, const __m128d *a, const __m128d *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					out_min[i] = _mm_min_ps(a[i], b[i]);
					out_max[i] = _mm_max_ps(a[i], b[i]);
				}
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void clamp(__m128d *out, const __m128d *value, const __m128d *min, const __m128d *max) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_ps(_mm_max_ps(value[i], min[i]), max[i]);
			}
		};
	}
#endif

	SVM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<double, N, Align>
		struct native_data_type<simd_mask<double, detail::avec<N, Align>>> { using type = __m128d; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<double, N, Align>
		struct native_data_size<simd_mask<double, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<double, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128d, detail::align_data<double, N, 16>()> to_native_data(simd_mask<double, detail::avec<N, A>>
		                                                                                            &value) noexcept
		requires
		detail::x86_overload_sse<double, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128d, detail::align_data<double, N, 16>()> to_native_data(const simd_mask<double, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<double, N, A>;
	}
}

#endif