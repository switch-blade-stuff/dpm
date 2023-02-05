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
	static constexpr T dp_sincos[3] = {7.85398125648498535156e-1, 3.77489470793079817668e-8, 2.69515142907905952645e-15};
	template<std::floating_point T>
	static constexpr T fopi = T{4.0} / std::numbers::pi_v<T>;

	template<std::floating_point T>
	static constexpr T sincof[6] = {
			1.58962301576546568060e-10, -2.50507477628578072866e-8,
			2.75573136213857245213e-6, -1.98412698295895385996e-4,
			8.33333333332211858878e-3, -1.66666666666666307295e-1
	};
	template<std::floating_point T>
	static constexpr T coscof[6] = {
			-1.13585365213876817300e-11, 2.08757008419747316778e-9,
			-2.75573141792967388112e-7, 2.48015872888517045348e-5,
			-1.38888888888730564116e-3, 4.16666666666665929218e-2
	};
}