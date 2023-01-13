/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "../define.hpp"

#ifdef DPM_HAS_IEEE754

#ifndef DPM_USE_IMPORT

#include <numbers>
#include <limits>

#endif

namespace dpm::detail
{
	static const double fopi_f64 = 4.0 / std::numbers::pi_v<double>; /* 4 / Pi */
}

#endif