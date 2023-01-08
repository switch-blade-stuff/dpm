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
		static __m128 x86_maskzero_vector_f32(__m128 m, std::size_t i) noexcept
		{
			switch (N - i)
			{
#ifdef SVM_HAS_SSE4_1
				case 3: return _mm_blend_ps(m, _mm_setzero_ps(), 0b1000);
				case 2: return _mm_blend_ps(m, _mm_setzero_ps(), 0b1100);
				case 1: return _mm_blend_ps(m, _mm_setzero_ps(), 0b1110);
#else
					case 3: return _mm_and_ps(m, _mm_set_ps(0.0f, std::bit_cast<float>(0xffff'ffff), std::bit_cast<float>(0xffff'ffff), std::bit_cast<float>(0xffff'ffff)));
					case 2: return _mm_and_ps(m, _mm_set_ps(0.0f, 0.0f, std::bit_cast<float>(0xffff'ffff), std::bit_cast<float>(0xffff'ffff)));
					case 1: return _mm_and_ps(m, _mm_set_ps(0.0f, 0.0f, 0.0f, std::bit_cast<float>(0xffff'ffff)));
#endif
				default: return m;
			}
		}
		template<std::size_t N>
		static __m128 x86_maskone_vector_f32(__m128 m, std::size_t i) noexcept
		{
			switch (const auto fm = std::bit_cast<float>(0xffff'ffff); N - i)
			{
#ifdef SVM_HAS_SSE4_1
				case 3: return _mm_blend_ps(m, _mm_set1_ps(fm), 0b1000);
				case 2: return _mm_blend_ps(m, _mm_set1_ps(fm), 0b1100);
				case 1: return _mm_blend_ps(m, _mm_set1_ps(fm), 0b1110);
#else
					case 3: return _mm_or_ps(m, _mm_set_ps(fm, 0.0f, 0.0f, 0.0f));
					case 2: return _mm_or_ps(m, _mm_set_ps(fm, fm, 0.0f, 0.0f));
					case 1: return _mm_or_ps(m, _mm_set_ps(fm, fm, fm, 0.0f));
#endif
				default: return m;
			}
		}

		template<std::size_t N>
		struct x86_impl<bool, __m128, N>
		{
			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_from(const bool *src, __m128 *dst, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 4)
				{
					float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
					switch (N - i)
					{
						default: f3 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i + 3])); [[fallthrough]];
						case 3: f2 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i + 2])); [[fallthrough]];
						case 2: f1 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i]));
					}
					dst[i / 4] = _mm_set_ps(f3, f2, f1, f0);
				}
			}
			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_to(bool *dst, const __m128 *src, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 4)
					switch (const auto bits = _mm_movemask_ps(src[i / 4]); N - i)
					{
						default: dst[i + 3] = bits & 0b1000; [[fallthrough]];
						case 3: dst[i + 2] = bits & 0b0100; [[fallthrough]];
						case 2: dst[i + 1] = bits & 0b0010; [[fallthrough]];
						case 1: dst[i] = bits & 0b0001;
					}
			}

			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_from(const bool *src, __m128 *dst, const __m128 *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 4)
				{
					float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
					switch (N - i)
					{
						default: f3 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i + 3])); [[fallthrough]];
						case 3: f2 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i + 2])); [[fallthrough]];
						case 2: f1 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i + 1])); [[fallthrough]];
						case 1: f0 = std::bit_cast<float>(detail::extend_bool<std::int32_t>(src[i]));
					}
					dst[i / 4] = _mm_and_ps(_mm_set_ps(f3, f2, f1, f0), mask[i / 4]);
				}
			}
			template<std::size_t M, typename F>
			static SVM_SAFE_ARRAY void copy_to(bool *dst, const __m128 *src, const __m128 *mask, F) noexcept
			{
				for (std::size_t i = 0; i < N; i += 4)
				{
					const auto bits = _mm_movemask_ps(_mm_and_ps(src[i / 4], mask[i / 4]));
					switch (N - i)
					{
						default: dst[i + 3] = bits & 0b1000; [[fallthrough]];
						case 3: dst[i + 2] = bits & 0b0100; [[fallthrough]];
						case 2: dst[i + 1] = bits & 0b0010; [[fallthrough]];
						case 1: dst[i] = bits & 0b0001;
					}
				}
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void invert(__m128 *dst, const __m128 *src) noexcept
			{
				const auto mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_ps(src[i], mask);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void bit_and(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_and_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void bit_or(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_or_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void bit_xor(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_xor_ps(a[i], b[i]);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_eq(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
#ifdef SVM_HAS_SSE2
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_castps_si128(a[i]);
					const auto vb = _mm_castps_si128(b[i]);
					out[i] = _mm_castsi128_ps(_mm_cmpeq_epi32(va, vb));
				}
#else
				const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_and_ps(a[i], nan_mask);
					const auto vb = _mm_and_ps(b[i], nan_mask);
					out[i] = _mm_cmpeq_ps(va, vb);
				}
#endif
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_ne(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
#ifdef SVM_HAS_SSE2
				const auto inv_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_castps_si128(a[i]);
					const auto vb = _mm_castps_si128(b[i]);
					out[i] = _mm_xor_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(va, vb)), inv_mask);
				}
#else
				const auto nan_mask = _mm_set1_ps(std::bit_cast<float>(0x3fff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto va = _mm_and_ps(a[i], nan_mask);
					const auto vb = _mm_and_ps(b[i], nan_mask);
					out[i] = _mm_cmpneq_ps(va, vb);
				}
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY bool all_of(const __m128 *mask) noexcept
			{
#ifdef SVM_HAS_SSE4_1
				if constexpr (M == 1)
				{
					const auto vm = x86_maskone_vector_f32<N>(mask[0], 0);
					return _mm_test_all_ones(_mm_castps_si128(vm));
				}
#endif
				auto result = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskone_vector_f32<N>(mask[i], i * 4);
					result = _mm_and_ps(result, vm);
				}
				return _mm_movemask_ps(result) == 0b1111;
			}
			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY bool any_of(const __m128 *mask) noexcept
			{
				auto result = _mm_setzero_ps();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f32<N>(mask[i], i * 4);
					result = _mm_or_ps(result, vm);
				}
#ifdef SVM_HAS_SSE4_1
				const auto vi = _mm_castps_si128(result);
				return !_mm_testz_si128(vi, vi);
#else
				return _mm_movemask_ps(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY bool none_of(const __m128 *mask) noexcept
			{
				auto result = _mm_setzero_ps();
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f32<N>(mask[i], i * 4);
					result = _mm_or_ps(result, vm);
				}
#ifdef SVM_HAS_SSE4_1
				const auto vi = _mm_castps_si128(result);
				return _mm_testz_si128(vi, vi);
#else
				return !_mm_movemask_ps(result);
#endif
			}
			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY bool some_of(const __m128 *mask) noexcept
			{
				auto any_mask = _mm_setzero_ps(), all_mask = _mm_set1_ps(std::bit_cast<float>(0xffff'ffff));
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = mask[i];
					const auto vmz = x86_maskzero_vector_f32<N>(vm, i * 4);
					const auto vmo = x86_maskone_vector_f32<N>(vm, i * 4);

					all_mask = _mm_and_ps(all_mask, vmo);
					any_mask = _mm_or_ps(any_mask, vmz);
				}
#ifdef SVM_HAS_SSE4_1
				const auto any_vi = _mm_castps_si128(any_mask);
				const auto all_vi = _mm_castps_si128(all_mask);
				return !_mm_testz_si128(any_vi, any_vi) && !_mm_test_all_ones(all_vi);
#else
				return _mm_movemask_ps(any_mask) && _mm_movemask_ps(all_mask) != 0b1111;
#endif
			}

			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY std::size_t popcount(const __m128 *mask) noexcept
			{
				std::size_t result = 0;
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto vm = x86_maskzero_vector_f32<N>(mask[i], i * 4);
					result += std::popcount(static_cast<std::uint32_t>(_mm_movemask_ps(vm)));
				}
				return result;
			}
			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY std::size_t find_first_set(const __m128 *mask) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					const auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(mask[i]));
					if (bits) return std::countr_zero(bits) + i * 4;
				}
				SVM_UNREACHABLE();
			}
			template<std::size_t M>
			[[nodiscard]] static SVM_SAFE_ARRAY std::size_t find_last_set(const __m128 *mask) noexcept
			{
				for (std::size_t i = M, k; (k = i--) != 0;)
				{
					auto bits = static_cast<std::uint8_t>(_mm_movemask_ps(mask[i]));
					switch (N - i * 4)
					{
						case 1: bits <<= 1; [[fallthrough]];
						case 2: bits <<= 1; [[fallthrough]];
						case 3: bits <<= 1; [[fallthrough]];
						default: bits <<= 4;
					}
					if (bits) return (k * 4 - 1) - std::countl_zero(bits);
				}
				SVM_UNREACHABLE();
			}

#ifdef SVM_HAS_SSE4_1
			template<std::size_t M>
			static SVM_SAFE_ARRAY void blend(__m128 *out, const __m128 *a, const __m128 *b, const __m128 *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_ps(a[i], b[i], m[i]);
			}
#endif
		};

		template<std::size_t N>
		struct x86_impl<float, __m128, N>
		{
			template<typename U>
			static U &data_at(__m128 *data, std::size_t i) noexcept { return reinterpret_cast<U *>(data)[i]; }
			template<typename U>
			static std::add_const_t<U> &data_at(const __m128 *data, std::size_t i) noexcept { return reinterpret_cast<std::add_const_t<U> *>(data)[i]; }

			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_from(const U *src, __m128 *dst, F) noexcept
			{
				if constexpr (std::same_as<U, float> && aligned_tag<F, alignof(__m128)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: dst[i / 4] = reinterpret_cast<const __m128 *>(src)[i / 4];
								break;
							case 3: data_at<float>(dst, i + 2) = src[i + 2]; [[fallthrough]];
							case 2: data_at<float>(dst, i + 1) = src[i + 1]; [[fallthrough]];
							case 1: data_at<float>(dst, i) = src[i];
						}
				}
#ifdef SVM_HAS_SSE2
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: dst[i / 4] = _mm_cvtepi32_ps(reinterpret_cast<__m128i *>(src)[i / 4]);
								break;
							case 3: data_at<float>(dst, i + 2) = static_cast<float>(src[i + 2]); [[fallthrough]];
							case 2: data_at<float>(dst, i + 1) = static_cast<float>(src[i + 1]); [[fallthrough]];
							case 1: data_at<float>(dst, i) = static_cast<float>(src[i]);
						}
				}
#endif
				else
					std::copy_n(src, N, reinterpret_cast<float *>(dst));
			}
			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_to(U *dst, const __m128 *src, F) noexcept
			{
				if constexpr (std::same_as<U, float> && aligned_tag<F, alignof(__m128)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: reinterpret_cast<__m128 *>(dst)[i / 4] = src[i / 4];
								break;
							case 3: dst[i + 2] = data_at<float>(src, i + 2); [[fallthrough]];
							case 2: dst[i + 1] = data_at<float>(src, i + 1); [[fallthrough]];
							case 1: dst[i] = data_at<float>(src, i);
						}
				}
#ifdef SVM_HAS_SSE2
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128)>)
				{
					for (std::size_t i = 0; i < N; i += 4)
						switch (N - i)
						{
							default: reinterpret_cast<__m128i *>(dst)[i / 4] = _mm_cvtps_epi32(src[i / 4]);
								break;
							case 3: dst[i + 2] = static_cast<std::int32_t>(data_at<float>(src, i + 2)); [[fallthrough]];
							case 2: dst[i + 1] = static_cast<std::int32_t>(data_at<float>(src, i + 1)); [[fallthrough]];
							case 1: dst[i] = static_cast<std::int32_t>(data_at<float>(src, i));
						}
				}
#endif
				else
					std::copy_n(reinterpret_cast<const float *>(src), N, dst);
			}

			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_from(const U *src, __m128 *dst, const __m128 *mask, F) noexcept
			{
#ifdef SVM_HAS_AVX
				if constexpr (std::same_as<U, float> && aligned_tag<F, alignof(__m128)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						dst[i] = _mm_maskload_ps(src + i, mi);
					}
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						dst[i] = _mm_cvtepi32_ps(_mm_maskload_epi32(src + i, mi));
					}
				else
#endif
					for (std::size_t i = 0; i < N; ++i) if (data_at<std::uint32_t>(mask, i)) data_at<float>(dst, i) = src[i];
			}
			template<std::size_t M, typename U, typename F>
			static SVM_SAFE_ARRAY void copy_to(U *dst, const __m128 *src, const __m128 *mask, F) noexcept
			{
#ifdef SVM_HAS_AVX
				if constexpr (std::same_as<U, float> && aligned_tag<F, alignof(__m128)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						_mm_maskstore_ps(dst + i, mi, src[i / 4]);
					}
				else if constexpr (std::same_as<U, std::int32_t> && aligned_tag<F, alignof(__m128)>)
					for (std::size_t i = 0; i < N; i += 4)
					{
						const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
						_mm_maskstore_epi32(dst + i, mi, _mm_cvtps_epi32(src[i / 4]));
					}
				else
#endif
				{
#ifdef SVM_HAS_SSE2
					if constexpr (std::same_as<U, float>)
						for (std::size_t i = 0; i < N; i += 4)
						{
							const auto mi = _mm_castps_si128(x86_maskzero_vector_f32<N>(mask[i / 4], i));
							const auto vi = _mm_castps_si128(src[i / 4]);
							_mm_maskmoveu_si128(vi, mi, reinterpret_cast<char *>(dst + i));
						}
					else
#endif
						for (std::size_t i = 0; i < N; ++i) if (data_at<std::uint32_t>(mask, i)) dst[i] = data_at<float>(src, i);
				}
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void inc(__m128 *data) noexcept
			{
				const auto one = _mm_set1_ps(1.0f);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_add_ps(data[i], one);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void dec(__m128 *data) noexcept
			{
				const auto one = _mm_set1_ps(1.0f);
				for (std::size_t i = 0; i < M; ++i) data[i] = _mm_sub_ps(data[i], one);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void negate(__m128 *dst, const __m128 *src) noexcept
			{
				const auto mask = _mm_set1_ps(std::bit_cast<float>(0x8000'0000));
				for (std::size_t i = 0; i < M; ++i) dst[i] = _mm_xor_ps(src[i], mask);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void add(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_add_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void sub(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_sub_ps(a[i], b[i]);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void mul(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_mul_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void div(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_div_ps(a[i], b[i]);
			}

			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_eq(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpeq_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_le(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmple_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_ge(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpge_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_lt(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmplt_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_gt(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpgt_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void cmp_ne(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_cmpneq_ps(a[i], b[i]);
			}

#ifdef SVM_HAS_SSE4_1
			template<std::size_t M>
			static SVM_SAFE_ARRAY void blend(__m128 *out, const __m128 *a, const __m128 *b, const __m128 *m) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_blendv_ps(a[i], b[i], m[i]);
			}
#endif

			template<std::size_t M>
			static SVM_SAFE_ARRAY void min(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void max(__m128 *out, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_max_ps(a[i], b[i]);
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void minmax(__m128 *out_min, __m128 *out_max, const __m128 *a, const __m128 *b) noexcept
			{
				for (std::size_t i = 0; i < M; ++i)
				{
					out_min[i] = _mm_min_ps(a[i], b[i]);
					out_max[i] = _mm_max_ps(a[i], b[i]);
				}
			}
			template<std::size_t M>
			static SVM_SAFE_ARRAY void clamp(__m128 *out, const __m128 *value, const __m128 *min, const __m128 *max) noexcept
			{
				for (std::size_t i = 0; i < M; ++i) out[i] = _mm_min_ps(_mm_max_ps(value[i], min[i]), max[i]);
			}
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<float, N, Align>
		struct native_data_type<simd_mask<float, detail::avec<N, Align>>> { using type = __m128; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<float, N, Align>
		struct native_data_size<simd_mask<float, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<float, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd_mask<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd_mask<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_sse<float, N, Align>
	class simd_mask<float, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd_mask>;

		using impl_t = detail::x86_impl<bool, __m128, detail::avec<N, Align>::size>;
		using vector_type = __m128;

		constexpr static auto data_size = ext::native_data_size_v<simd_mask>;
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];

	public:
		using value_type = bool;
		using reference = detail::mask_reference<std::int32_t>;

		using abi_type = detail::avec<N, Align>;
		using simd_type = simd<float, abi_type>;

		/** Returns width of the SIMD mask. */
		static constexpr std::size_t size() noexcept { return abi_type::size; }

	public:
		constexpr simd_mask() noexcept = default;
		constexpr simd_mask(const simd_mask &) noexcept = default;
		constexpr simd_mask &operator=(const simd_mask &) noexcept = default;
		constexpr simd_mask(simd_mask &&) noexcept = default;
		constexpr simd_mask &operator=(simd_mask &&) noexcept = default;

		/** Initializes the SIMD mask object with a native SSE vector.
		 * @note This constructor is available for overload resolution only when the SIMD mask contains a single SSE vector. */
		constexpr SVM_SAFE_ARRAY simd_mask(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD mask object with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd_mask) / sizeof(__m128)`. */
		constexpr SVM_SAFE_ARRAY simd_mask(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		constexpr SVM_SAFE_ARRAY simd_mask(value_type value) noexcept
		{
			const auto v = value ? _mm_set1_ps(std::bit_cast<float>(0xffff'ffff)) : _mm_setzero_ps();
			std::fill_n(m_data, data_size, v);
		}
		/** Copies elements from \a other. */
		template<typename U, std::size_t OtherAlign>
		SVM_SAFE_ARRAY simd_mask(const simd_mask<U, detail::avec<size(), OtherAlign>> &other) noexcept
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
		void copy_from(const value_type *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_from<data_size>(mem, m_data, Flags{}); }
		/** Copies the underlying elements to \a mem. */
		template<typename Flags>
		void copy_to(value_type *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_to<data_size>(mem, m_data, Flags{}); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			SVM_ASSERT(i < size());
			return reference{reinterpret_cast<std::int32_t *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			SVM_ASSERT(i < size());
			return reinterpret_cast<const std::int32_t *>(m_data)[i];
		}

		[[nodiscard]] SVM_SAFE_ARRAY simd_mask operator!() const noexcept
		{
			simd_mask result;
			impl_t::template invert<data_size>(result.m_data, m_data);
			return result;
		}

		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator|(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator^(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_xor<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend SVM_SAFE_ARRAY simd_mask &operator&=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_and<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend SVM_SAFE_ARRAY simd_mask &operator|=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_or<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend SVM_SAFE_ARRAY simd_mask &operator^=(simd_mask &a, const simd_mask &b) noexcept
		{
			impl_t::template bit_xor<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator&&(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_and<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator||(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template bit_or<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator==(const simd_mask &a, const simd_mask &b) noexcept
		{
			simd_mask result;
			impl_t::template cmp_eq<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY simd_mask operator!=(const simd_mask &a, const simd_mask &b) noexcept
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
		template<std::size_t N, std::size_t A> requires detail::x86_overload_sse<float, N, A>
		struct native_access<simd_mask<float, avec<N, A>>>
		{
			using mask_t = simd_mask<float, avec<N, A>>;

			[[nodiscard]] static std::span<__m128, mask_t::data_size> to_native_data(mask_t &value) noexcept { return {value.m_data}; }
			[[nodiscard]] static std::span<const __m128, mask_t::data_size> to_native_data(const mask_t &value) noexcept { return {value.m_data}; }
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd_mask<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>
		{
			return detail::native_access<simd_mask<float, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a constant span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd_mask<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>
		{
			return detail::native_access<simd_mask<float, detail::avec<N, A>>>::to_native_data(value);
		}

#ifdef SVM_HAS_SSE4_1
		/** Replaces elements of mask \a a with elements of mask \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd_mask<float, detail::avec<N, A>> blend(
				const simd_mask<float, detail::avec<N, A>> &a,
				const simd_mask<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_sse<float, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd_mask<float, detail::avec<N, A>>>;

			simd_mask<float, detail::avec<N, A>> result;
			detail::x86_impl<bool, __m128, N>::template blend<data_size>(
					to_native_data(result).data(),
					to_native_data(a).data(),
					to_native_data(b).data(),
					to_native_data(m).data()
			);
			return result;
		}
#endif
	}

	/** Returns `true` if all of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool all_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template all_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask are `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool any_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template any_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at none of the elements of the \a mask is `true`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool none_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template none_of<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns `true` if at least one of the elements of the \a mask is `true` and at least one is `false`. Otherwise returns `false`. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline bool some_of(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template some_of<data_size>(ext::to_native_data(mask).data());
	}

	/** Returns the number of `true` elements of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t popcount(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template popcount<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the first `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_first_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template find_first_set<data_size>(ext::to_native_data(mask).data());
	}
	/** Returns the index of the last `true` element of \a mask. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::size_t find_last_set(const simd_mask<float, detail::avec<N, A>> &mask) noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd_mask<float, detail::avec<N, A>>>;
		return detail::x86_impl<bool, __m128, N>::template find_last_set<data_size>(ext::to_native_data(mask).data());
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<float, N, Align>
		struct native_data_type<simd<float, detail::avec<N, Align>>> { using type = __m128; };
		template<std::size_t N, std::size_t Align> requires simd_abi::detail::x86_overload_sse<float, N, Align>
		struct native_data_size<simd<float, detail::avec<N, Align>>> : std::integral_constant<std::size_t, detail::align_data<float, N, 16>()> {};

		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>;
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>;
	}

	template<std::size_t N, std::size_t Align> requires detail::x86_overload_sse<float, N, Align>
	class simd<float, detail::avec<N, Align>>
	{
		friend struct detail::native_access<simd>;

		using impl_t = detail::x86_impl<float, __m128, detail::avec<N, Align>::size>;
		using vector_type = __m128;

		constexpr static auto data_size = ext::native_data_size_v<simd>;
		constexpr static auto alignment = std::max(Align, alignof(vector_type));

		using data_type = vector_type[data_size];

	public:
		using value_type = float;
		using reference = detail::simd_reference<value_type>;

		using abi_type = detail::avec<N, Align>;
		using mask_type = simd_mask<float, abi_type>;

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
		constexpr SVM_SAFE_ARRAY simd(__m128 native) noexcept requires (data_size == 1) { m_data[0] = native; }
		/** Initializes the SIMD vector with an array of native SSE vectors.
		 * @note Size of the native vector array must be the same as `sizeof(simd) / sizeof(__m128)`. */
		constexpr SVM_SAFE_ARRAY simd(const __m128 (&native)[data_size]) noexcept { std::copy_n(native, data_size, m_data); }

		/** Initializes the underlying elements with \a value. */
		template<detail::compatible_element<value_type> U>
		SVM_SAFE_ARRAY simd(U &&value) noexcept
		{
			const auto vec = _mm_set1_ps(static_cast<float>(value));
			std::fill_n(m_data, data_size, vec);
		}
		/** Initializes the underlying elements with values provided by the generator \a gen. */
		template<detail::element_generator<value_type, size()> G>
		SVM_SAFE_ARRAY simd(G &&gen) noexcept
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
		SVM_SAFE_ARRAY simd(const simd<U, detail::avec<size(), OtherAlign>> &other) noexcept
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
		void copy_from(const U *mem, Flags) noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_from<data_size>(mem, m_data, Flags{}); }
		/** Copies the underlying elements to \a mem. */
		template<typename U, typename Flags>
		void copy_to(U *mem, Flags) const noexcept requires is_simd_flag_type_v<Flags> { impl_t::template copy_to<data_size>(mem, m_data, Flags{}); }

		[[nodiscard]] reference operator[](std::size_t i) noexcept
		{
			SVM_ASSERT(i < size());
			return reference{reinterpret_cast<float *>(m_data)[i]};
		}
		[[nodiscard]] value_type operator[](std::size_t i) const noexcept
		{
			SVM_ASSERT(i < size());
			return reinterpret_cast<const float *>(m_data)[i];
		}

		SVM_SAFE_ARRAY simd operator++(int) noexcept
		{
			auto tmp = *this;
			operator++();
			return tmp;
		}
		SVM_SAFE_ARRAY simd operator--(int) noexcept
		{
			auto tmp = *this;
			operator--();
			return tmp;
		}
		inline SVM_SAFE_ARRAY simd &operator++() noexcept
		{
			impl_t::template inc<data_size>(m_data);
			return *this;
		}
		inline SVM_SAFE_ARRAY simd &operator--() noexcept
		{
			impl_t::template inc<data_size>(m_data);
			return *this;
		}

		[[nodiscard]] simd SVM_SAFE_ARRAY operator+() const noexcept { return *this; }
		[[nodiscard]] simd SVM_SAFE_ARRAY operator-() const noexcept
		{
			simd result;
			impl_t::template negate<data_size>(result.m_data, m_data);
			return result;
		}

		[[nodiscard]] friend SVM_SAFE_ARRAY simd operator+(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template add<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY simd operator-(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template sub<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend SVM_SAFE_ARRAY simd &operator+=(simd &a, const simd &b) noexcept
		{
			impl_t::template add<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend SVM_SAFE_ARRAY simd &operator-=(simd &a, const simd &b) noexcept
		{
			impl_t::template sub<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend SVM_SAFE_ARRAY simd operator*(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template mul<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY simd operator/(const simd &a, const simd &b) noexcept
		{
			simd result;
			impl_t::template div<data_size>(result.m_data, a.m_data, b.m_data);
			return result;
		}

		friend SVM_SAFE_ARRAY simd &operator*=(simd &a, const simd &b) noexcept
		{
			impl_t::template mul<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}
		friend SVM_SAFE_ARRAY simd &operator/=(simd &a, const simd &b) noexcept
		{
			impl_t::template div<data_size>(a.m_data, a.m_data, b.m_data);
			return a;
		}

		[[nodiscard]] friend SVM_SAFE_ARRAY mask_type operator==(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_eq<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY mask_type operator<=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_le<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY mask_type operator>=(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_ge<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY mask_type operator<(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_lt<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY mask_type operator>(const simd &a, const simd &b) noexcept
		{
			data_type mask_data;
			impl_t::template cmp_gt<data_size>(mask_data, a.m_data, b.m_data);
			return {mask_data};
		}
		[[nodiscard]] friend SVM_SAFE_ARRAY mask_type operator!=(const simd &a, const simd &b) noexcept
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
		template<std::size_t N, std::size_t A> requires detail::x86_overload_sse<float, N, A>
		struct native_access<simd<float, avec<N, A>>>
		{
			using simd_t = simd<float, avec<N, A>>;

			[[nodiscard]] static std::span<__m128, simd_t::data_size> to_native_data(simd_t &value) noexcept { return {value.m_data}; }
			[[nodiscard]] static std::span<const __m128, simd_t::data_size> to_native_data(const simd_t &value) noexcept { return {value.m_data}; }
		};
	}

	SVM_DECLARE_EXT_NAMESPACE
	{
		/** Returns a span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<__m128, detail::align_data<float, N, 16>()> to_native_data(simd<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>
		{
			return detail::native_access<simd<float, detail::avec<N, A>>>::to_native_data(value);
		}
		/** Returns a constant span of the underlying SSE vectors for \a value. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline std::span<const __m128, detail::align_data<float, N, 16>()> to_native_data(const simd<float, detail::avec<N, A>> &value) noexcept requires detail::x86_overload_sse<float, N, A>
		{
			return detail::native_access<simd<float, detail::avec<N, A>>>::to_native_data(value);
		}

#ifdef SVM_HAS_SSE4_1
		/** Replaces elements of vector \a a with elements of vector \a b using mask \a m. Elements of \a b are selected if the corresponding element of \a m evaluates to `true`. */
		template<std::size_t N, std::size_t A>
		[[nodiscard]] inline simd<float, detail::avec<N, A>> blend(
				const simd<float, detail::avec<N, A>> &a,
				const simd<float, detail::avec<N, A>> &b,
				const simd_mask<float, detail::avec<N, A>> &m)
		noexcept requires detail::x86_overload_sse<float, N, A>
		{
			constexpr auto data_size = native_data_size_v<simd<float, detail::avec<N, A>>>;

			simd<float, detail::avec<N, A>> result;
			detail::x86_impl<float, __m128, N>::template blend<data_size>(
					to_native_data(result).data(),
					to_native_data(a).data(),
					to_native_data(b).data(),
					to_native_data(m).data()
			);
			return result;
		}
#endif
	}

	/** Returns an SIMD vector of minimum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<float, detail::avec<N, A>> min(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result;
		detail::x86_impl<float, __m128, N>::template min<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}
	/** Returns an SIMD vector of maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<float, detail::avec<N, A>> max(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result;
		detail::x86_impl<float, __m128, N>::template max<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}

	/** Returns a pair of SIMD vectors of minimum and maximum elements of \a a and \a b. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline std::pair<simd<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> minmax(
			const simd<float, detail::avec<N, A>> &a,
			const simd<float, detail::avec<N, A>> &b)
	noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		std::pair<simd<float, detail::avec<N, A>>, simd<float, detail::avec<N, A>>> result;
		detail::x86_impl<float, __m128, N>::template minmax<data_size>(
				ext::to_native_data(result.first).data(),
				ext::to_native_data(result.second).data(),
				ext::to_native_data(a).data(),
				ext::to_native_data(b).data()
		);
		return result;
	}

	/** Clamps elements \f \a value between corresponding elements of \a ming and \a max. */
	template<std::size_t N, std::size_t A>
	[[nodiscard]] inline simd<float, detail::avec<N, A>> clamp(
			const simd<float, detail::avec<N, A>> &value,
			const simd<float, detail::avec<N, A>> &min,
			const simd<float, detail::avec<N, A>> &max)
	noexcept requires detail::x86_overload_sse<float, N, A>
	{
		constexpr auto data_size = ext::native_data_size_v<simd<float, detail::avec<N, A>>>;

		simd<float, detail::avec<N, A>> result;
		detail::x86_impl<float, __m128, N>::template clamp<data_size>(
				ext::to_native_data(result).data(),
				ext::to_native_data(value).data(),
				ext::to_native_data(min).data(),
				ext::to_native_data(max).data()
		);
		return result;
	}
}

#endif