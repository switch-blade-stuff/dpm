/*
 * Created by switchblade on 2023-01-03.
 */

#pragma once

#include "define.hpp"

#if !defined(NDEBUG) || defined(SVM_DEBUG)

#ifdef NDEBUG
#define SVM_ENABLE_NDEBUG
#undef NDEBUG
#endif

#include <cassert>

#ifdef SVM_ENABLE_NDEBUG
#undef SVM_ENABLE_NDEBUG
#define NDEBUG
#endif

#define SVM_ASSERT(cnd) assert(cnd); SVM_ASSUME(cnd)
#else
#define SVM_ASSERT(cnd) SVM_ASSUME(cnd)
#endif
