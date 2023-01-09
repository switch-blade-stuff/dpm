/*
 * Created by switchblade on 2023-01-03.
 */

#pragma once

#include "define.hpp"

#if !defined(NDEBUG) || defined(DPM_DEBUG)

#ifdef NDEBUG
#define DPM_ENABLE_NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#ifdef DPM_ENABLE_NDEBUG
#undef DPM_ENABLE_NDEBUG
#define NDEBUG
#endif

#define DPM_ASSERT(cnd) assert(cnd); DPM_ASSUME(cnd)
#else
#define DPM_ASSERT(cnd) DPM_ASSUME(cnd)
#endif
