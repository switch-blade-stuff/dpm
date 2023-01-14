/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "../../../define.hpp"

#if defined(DPM_ARCH_X86) && (defined(DPM_HAS_SSE2) || defined(DPM_DYNAMIC_DISPATCH))

#include <emmintrin.h>

namespace dpm::detail
{
	[[nodiscard]] std::pair<__m128d, __m128d> DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_sincos(__m128d x) noexcept;
	[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_sin(__m128d x) noexcept;
	[[nodiscard]] __m128d DPM_PUBLIC DPM_PURE DPM_VECTORCALL DPM_TARGET("sse2") x86_cos(__m128d x) noexcept;
}

#endif