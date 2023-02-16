/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "mbase.hpp"


namespace dpm
{
	namespace detail
	{
#ifdef DPM_USE_SVML
		[[nodiscard]] DPM_FORCEINLINE __m128 exp(__m128 x) noexcept { return _mm_exp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 exp2(__m128 x) noexcept { return _mm_exp2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 expm1(__m128 x) noexcept { return _mm_expm1_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m128 log(__m128 x) noexcept { return _mm_log_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 log2(__m128 x) noexcept { return _mm_log2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 log10(__m128 x) noexcept { return _mm_log10_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 log1p(__m128 x) noexcept { return _mm_log1p_ps(x); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d exp(__m128d x) noexcept { return _mm_exp_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d exp2(__m128d x) noexcept { return _mm_exp2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d expm1(__m128d x) noexcept { return _mm_expm1_pd(x); }

		[[nodiscard]] DPM_FORCEINLINE __m128d log(__m128d x) noexcept { return _mm_log_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d log2(__m128d x) noexcept { return _mm_log2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d log10(__m128d x) noexcept { return _mm_log10_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d log1p(__m128d x) noexcept { return _mm_log1p_pd(x); }
#endif
#ifdef DPM_HAS_AVX
		[[nodiscard]] DPM_FORCEINLINE __m256 exp(__m256 x) noexcept { return _mm256_exp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 exp2(__m256 x) noexcept { return _mm256_exp2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 expm1(__m256 x) noexcept { return _mm256_expm1_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256 log(__m256 x) noexcept { return _mm256_log_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 log2(__m256 x) noexcept { return _mm256_log2_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 log10(__m256 x) noexcept { return _mm256_log10_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256 log1p(__m256 x) noexcept { return _mm256_log1p_ps(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d exp(__m256d x) noexcept { return _mm256_exp_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d exp2(__m256d x) noexcept { return _mm256_exp2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d expm1(__m256d x) noexcept { return _mm256_expm1_pd(x); }

		[[nodiscard]] DPM_FORCEINLINE __m256d log(__m256d x) noexcept { return _mm256_log_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d log2(__m256d x) noexcept { return _mm256_log2_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d log10(__m256d x) noexcept { return _mm256_log10_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m256d log1p(__m256d x) noexcept { return _mm256_log1p_pd(x); }
#endif
#else
		enum exp_op
		{
			OP_LOG,
			OP_LOG2,
			OP_LOG10,
			OP_LOG1P
		};

		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log2(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log10(__m128 x) noexcept;
		[[nodiscard]] __m128 DPM_PUBLIC DPM_MATHFUNC log1p(__m128 x) noexcept;

#ifdef DPM_HAS_SSE2
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log2(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log10(__m128d x) noexcept;
		[[nodiscard]] __m128d DPM_PUBLIC DPM_MATHFUNC log1p(__m128d x) noexcept;
#endif
#ifdef DPM_HAS_AVX
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log2(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log10(__m256 x) noexcept;
		[[nodiscard]] __m256 DPM_PUBLIC DPM_MATHFUNC log1p(__m256 x) noexcept;

		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log2(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log10(__m256d x) noexcept;
		[[nodiscard]] __m256d DPM_PUBLIC DPM_MATHFUNC log1p(__m256d x) noexcept;
#endif
#endif
	}
}
