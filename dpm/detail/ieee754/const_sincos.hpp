/*
 * Created by switchblade on 2023-01-10.
 */

#pragma once

#include "../define.hpp"

#ifdef DPM_HAS_IEEE754

#ifndef DPM_USE_IMPORT

#include <array>

#endif

namespace dpm::detail
{
	extern const DPM_PRIVATE std::array<double, 440> sincos_tbl_f64;
}

#endif