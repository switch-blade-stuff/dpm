/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "export.gen.hpp"
#include "arch.hpp"

#include <version>

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

#if defined(DPM_EXPORT)
#define DPM_API_PUBLIC DPM_API_EXPORT
#else
#define DPM_API_PUBLIC DPM_API_IMPORT
#endif

#if defined(_MSC_VER)
#define DPM_FUNCNAME __FUNCSIG__
#define DPM_FORCEINLINE inline __forceinline
#define DPM_NEVER_INLINE __declspec(noinline)
#define DPM_PURE

/* Windows calling convention will never use vector registers for function arguments unless explicitly required. */
#define DPM_VECTORCALL __vectorcall

#elif defined(__clang__) || defined(__GNUC__)
#define DPM_FUNCNAME __PRETTY_FUNCTION__
#define DPM_FORCEINLINE inline __attribute__((always_inline))
#define DPM_NEVER_INLINE __attribute__((noinline))
#define DPM_PURE __attribute__((pure))
#define DPM_VECTORCALL
#else
#define DPM_FUNCNAME __func__
#define DPM_FORCEINLINE inline
#define DPM_NEVER_INLINE
#define DPM_VECTORCALL
#define DPM_PURE
#endif

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

/* Modules are currently not supported */
//#if defined(DPM_USE_MODULES) && defined(__cpp_modules)
//#define DPM_USE_IMPORT
//#endif

#ifdef DPM_INLINE_EXTENSIONS
#define DPM_DECLARE_EXT_NAMESPACE inline namespace ext
#else
#define DPM_DECLARE_EXT_NAMESPACE namespace ext
#endif

/* DPM_HANDLE_ERRORS implies DPM_PROPAGATE_NAN */
#if defined(DPM_HANDLE_ERRORS) && !defined(DPM_PROPAGATE_NAN)
#define DPM_PROPAGATE_NAN
#endif