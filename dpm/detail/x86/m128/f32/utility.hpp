/*
 * Created by switchblade on 2023-01-12.
 */

#pragma once

#include "../../fwd.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE) || defined(DPM_DYNAMIC_DISPATCH))

namespace dpm::detail
{
#ifdef DPM_HAS_SSE2
	[[nodiscard]] inline DPM_FORCEINLINE __m128d x86_cvt_u32_f32(__m128i x) noexcept
	{
		const auto a = _mm_cvtepi32_ps(_mm_and_si128(x, _mm_set1_epi32(1)));
		const auto b = _mm_cvtepi32_ps(_mm_srli_epi32(x, 1));
		return _mm_add_ps(_mm_add_ps(b, b), a); /* (x >> 1) * 2 + x & 1 */
	}
	[[nodiscard]] inline DPM_FORCEINLINE __m128i x86_cvt_f32_u32(__m128d x) noexcept
	{
		const auto offset = _mm_set1_ps(0x40'0000);
		return _mm_xor_si128(std::bit_cast<__m128i>(_mm_add_ps(x, offset)), std::bit_cast<__m128i>(offset));
	}
#endif
}

#endif