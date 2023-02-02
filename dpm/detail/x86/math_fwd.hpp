/*
 * Created by switchblade on 2023-02-01.
 */

#pragma once

#include "../define.hpp"

#define DPM_MATHFUNC(tgt) DPM_PURE DPM_VECTORCALL DPM_TARGET(tgt)

#ifdef DPM_HANDLE_ERRORS
#ifndef DPM_USE_IMPORT

#include <cerrno>
#include <cfenv>

#endif
#endif