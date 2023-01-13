/*
 * Created by switchblade on 2023-01-12.
 */

#include "../../../define.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include <smmintrin.h>

namespace dpm::detail
{
#ifndef DPM_HAS_SSE4_1
	[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_floor_f64(__m128i x) noexcept;
#else
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_floor_f64(__m128i x) noexcept { return _mm_floor_pd(x); }
#endif

#ifndef DPM_HAS_AVX512DQ
	[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_cvt_u64_f64(__m128i x) noexcept;
	[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_cvt_i64_f64(__m128i x) noexcept;
	[[nodiscard]] __m128i DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_cvt_f64_u64(__m128d x) noexcept;
	[[nodiscard]] __m128i DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_cvt_f64_i64(__m128d x) noexcept;
#else
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_cvt_u64_f64(__m128i x) noexcept { return _mm_cvtepu64_pd(x); }
	[[nodiscard]] inline __m128d DPM_FORCEINLINE x86_cvt_i64_f64(__m128i x) noexcept { return _mm_cvtepi64_pd(x); }
	[[nodiscard]] inline __m128i DPM_FORCEINLINE x86_cvt_f64_u64(__m128d x) noexcept { return _mm_cvtpd_epu64(x); }
	[[nodiscard]] inline __m128i DPM_FORCEINLINE x86_cvt_f64_i64(__m128d x) noexcept { return _mm_cvtpd_epi64(x); }
#endif
}

#endif