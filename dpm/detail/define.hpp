/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

/* Modules are currently not supported */
#if defined(DPM_USE_MODULES) && defined(__cpp_modules) && 0
#define DPM_USE_IMPORT
#endif

#include <type_traits>
#include <concepts>
#include <version>
#include <cstdint>
#include <cstddef>

#include "api.hpp"
#include "arch.hpp"
#include "../debug.hpp"

#if defined(_MSC_VER)

#define DPM_TARGET(t)
#define DPM_MAY_ALIAS

#elif defined(__clang__) || defined(__GNUC__)

#define DPM_TARGET(t) __attribute__((target(t)))
#define DPM_MAY_ALIAS __attribute__((__may_alias__))

#else

#define DPM_TARGET(t)
#define DPM_MAY_ALIAS

#endif

#ifdef DPM_INLINE_EXTENSIONS
#define DPM_DECLARE_EXT_NAMESPACE inline namespace ext
#else
#define DPM_DECLARE_EXT_NAMESPACE namespace ext
#endif

/* DPM_HANDLE_ERRORS implies DPM_PROPAGATE_NAN */
#if defined(DPM_HANDLE_ERRORS) && !defined(DPM_PROPAGATE_NAN)
#define DPM_PROPAGATE_NAN
#endif