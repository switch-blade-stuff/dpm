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
	static constexpr float dp_sincos_f32[3] = {7.85398125648e-1, 3.77489470793e-8, 2.69515147674e-15};
	static constexpr float fopi_f32 = 4.0 / std::numbers::pi_v<float>;

	static constexpr float sincof_f32[6] = {
			1.58962301655e-10, -2.50507472543e-8,
			2.75573142972e-6, -1.98412701138e-4,
			8.33333376795e-3, -1.66666671634e-1
	};
	static constexpr float coscof_f32[6] = {
			-1.13585365072e-11, 2.08757011677e-9,
			-2.75573142972e-7, 2.48015876423e-5,
			-1.38888892252e-3, 4.16666679084e-2
	};
}