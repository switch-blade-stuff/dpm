/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "define.hpp"

#ifndef DPM_USE_IMPORT

#include <numbers>
#include <limits>

#endif

namespace dpm::detail
{
	template<std::floating_point T>
	static constexpr T dp_sincos[] = {T{7.85398125648498535156e-1}, T{3.77489470793079817668e-8}, T{2.69515142907905952645e-15}};
	template<std::floating_point T>
	static constexpr T fopi = T{4.0} / std::numbers::pi_v<T>;

	template<std::floating_point T>
	static constexpr T sincof[] = {
			T{1.58962301576546568060e-10}, T{-2.50507477628578072866e-8},
			T{2.75573136213857245213e-6}, T{-1.98412698295895385996e-4},
			T{8.33333333332211858878e-3}, T{-1.66666666666666307295e-1}
	};
	template<std::floating_point T>
	static constexpr T coscof[] = {
			T{-1.13585365213876817300e-11}, T{2.08757008419747316778e-9},
			T{-2.75573141792967388112e-7}, T{2.48015872888517045348e-5},
			T{-1.38888888888730564116e-3}, T{4.16666666666665929218e-2}
	};

	template<std::floating_point T>
	static constexpr T dp_tancot[] = {T{-7.853981554508209228515625e-1L}, T{-7.853981554508209228515625e-9L}, T{-7.853981554508209228515625e-17L}};
	template<std::floating_point T>
	static constexpr T  tancot_q[] = {T{1.00000000000000000000e0L}, T{1.3681296347069295467845e4L}, T{-1.3208923444021096744731e6L}, T{2.5008380182335791583922e7L}, T{-5.3869575592945462988123e7L}};
	template<std::floating_point T>
	static constexpr T  tancot_p[] = {T{-1.3093693918138377764608e4L}, T{1.1535166483858741613983e6L}, T{-1.7956525197648487798769e7L}};

	template<std::floating_point T>
	static constexpr T tancot_pmin = std::same_as<T, float> ? T{1.0e-4} : std::same_as<T, double> ? T{1.0e-14} : T{1.0e-20L};
}