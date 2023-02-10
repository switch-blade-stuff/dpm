/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "../debug.hpp"

/* Modules are currently not supported */
#if defined(DPM_USE_MODULES) && defined(__cpp_modules) && 0
#define DPM_USE_IMPORT
#endif

#include <type_traits>
#include <concepts>
#include <version>
#include <cstdint>
#include <cstddef>

#include "arch.hpp"

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

#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)

#define DPM_API_HIDDEN
#if defined(_MSC_VER)
#define DPM_API_EXPORT __declspec(dllexport)
#define DPM_API_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#define DPM_API_EXPORT __attribute__((dllexport))
#define DPM_API_IMPORT __attribute__((dllimport))
#endif

#elif __GNUC__ >= 4

#define DPM_API_HIDDEN __attribute__((visibility("hidden")))
#define DPM_API_EXPORT __attribute__((visibility("default")))
#define DPM_API_IMPORT __attribute__((visibility("default")))

#else

#define DPM_API_HIDDEN
#define DPM_API_EXPORT
#define DPM_API_IMPORT

#endif

#ifdef DPM_EXPORT
#define DPM_PUBLIC DPM_API_EXPORT
#define DPM_PRIVATE DPM_API_HIDDEN
#else
#define DPM_PUBLIC DPM_API_IMPORT
#define DPM_PRIVATE DPM_API_HIDDEN
#endif

#ifdef DPM_INLINE_EXTENSIONS
#define DPM_DECLARE_EXT_NAMESPACE inline namespace ext
#else
#define DPM_DECLARE_EXT_NAMESPACE namespace ext
#endif