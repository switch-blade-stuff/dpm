/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

#include "mbase.hpp"

#ifdef DPM_USE_SVML

namespace dpm
{
	namespace detail
	{
		[[nodiscard]] DPM_FORCEINLINE __m128 exp(__m128 x) noexcept { return _mm_exp_ps(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128 exp2(__m128 x) noexcept { return _mm_exp2_ps(x); }

#ifdef DPM_HAS_SSE2
		[[nodiscard]] DPM_FORCEINLINE __m128d exp(__m128d x) noexcept { return _mm_exp_pd(x); }
		[[nodiscard]] DPM_FORCEINLINE __m128d exp2(__m128d x) noexcept { return _mm_exp2_pd(x); }
#endif
#ifdef DPM_HAS_AVX

#endif
	}
}

#endif