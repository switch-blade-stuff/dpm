/*
 * Created by switchblade on 2023-02-10.
 */

#ifdef DPM_HANDLE_ERRORS
#ifndef _MSC_VER /* MSVC does not support STDC pragmas */
#pragma STDC FENV_ACCESS ON
#endif
#endif

#include "exp.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE2) && !defined(DPM_USE_SVML)

namespace dpm::detail
{
	template<std::same_as<float> T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V impl_log(V x) noexcept
	{
		auto y = eval_log<T>(log_normalize<T>(x));
		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0f)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}
	template<std::same_as<double> T, typename V>
	[[nodiscard]] DPM_FORCEINLINE V impl_log(V x) noexcept
	{
		auto y = eval_log<T>(log_normalize<T>(x));
		/* log(1) == 0 */
		y = bit_andnot(cmp_eq<T>(x, fill<V>(1.0)), y);
		/* Handle negative x, NaN & inf. */
		return log_excepts<T>(y, x);
	}

	__m128 DPM_MATHFUNC log(__m128 x) noexcept { return impl_log<float>(x); }
	__m128d DPM_MATHFUNC log(__m128d x) noexcept { return impl_log<double>(x); }
#ifdef DPM_HAS_AVX
	__m256 DPM_MATHFUNC log(__m256 x) noexcept { return impl_log<float>(x); }
	__m256d DPM_MATHFUNC log(__m256d x) noexcept { return impl_log<double>(x); }
#endif
}

#endif