/*
 * Created by switchblade on 2023-02-01.
 */

#pragma once

#include "../define.hpp"

#define DPM_MATHFUNC DPM_PURE DPM_VECTORCALL

#ifdef DPM_HANDLE_ERRORS
#ifndef DPM_USE_IMPORT

#include <cerrno>
#include <cfenv>

#endif
#endif