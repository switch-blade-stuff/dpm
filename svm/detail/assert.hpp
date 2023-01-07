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

#if defined(_MSC_VER) || defined(__CYGWIN__)
#define SVM_ASSERT(cnd, msg) _assert("Assertion " #cnd " failed: " msg, __FILE__, __LINE__)
#else
#define SVM_ASSERT(cnd, msg) assert((cnd) && msg)
#endif

#else
#define SVM_ASSERT(cnd, msg)
#endif
