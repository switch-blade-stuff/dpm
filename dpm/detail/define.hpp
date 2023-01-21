/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "arch.hpp"

#if !defined(NDEBUG) && !defined(DPM_DEBUG)
#define DPM_DEBUG
#endif

/* Define DPM_USE_IMPORT only if modules are enabled and supported by the compiler. */
#if defined(DPM_USE_MODULES) && defined(__cpp_modules)
#define DPM_USE_IMPORT
#endif

#ifdef DPM_USE_IMPORT

/* If we are not on MSVC use `import std`. Otherwise, use `import std.core`. */
#ifdef _MSC_VER

import std.core;

#else

import std;

#endif

#else

#include <type_traits>
#include <concepts>
#include <version>
#include <cstdint>
#include <cstddef>

#endif

#if defined(__has_cpp_attribute) && __has_cpp_attribute(assume)
#define DPM_ASSUME(x)
#elif defined(_MSC_VER)
#define DPM_ASSUME(x) __assume(x)
#elif 0 && defined(__clang__) /* See https://github.com/llvm/llvm-project/issues/55636 and https://github.com/llvm/llvm-project/issues/45902 */
#define DPM_ASSUME(x) __builtin_assume(x)
#elif defined(__GNUC__)
#define DPM_ASSUME(x) if (!(x)) __builtin_unreachable()
#else
#define DPM_ASSUME(x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define DPM_UNREACHABLE() __builtin_unreachable()
#elif defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define DPM_UNREACHABLE() std::unreachable()
#else
#define DPM_UNREACHABLE() DPM_ASSUME(false)
#endif

#if defined(_MSC_VER)

#define DPM_PURE
#define DPM_TARGET(t)
#define DPM_FORCEINLINE __forceinline
#define DPM_NEVER_INLINE __declspec(noinline)

/* Windows calling convention will never use vector registers for function arguments. Force it. */
#define DPM_VECTORCALL __vectorcall

#elif defined(__clang__) || defined(__GNUC__)

#define DPM_PURE __attribute__((pure))
#define DPM_TARGET(t) __attribute__((target(t)))
#define DPM_FORCEINLINE __attribute__((always_inline))
#define DPM_NEVER_INLINE __attribute__((noinline))
#define DPM_VECTORCALL

#else

#define DPM_PURE
#define DPM_TARGET(t)
#define DPM_FORCEINLINE
#define DPM_NEVER_INLINE
#define DPM_VECTORCALL

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
#define DPM_PUBLIC DPM_API_IMPORT
#define DPM_PRIVATE DPM_API_HIDDEN
#else
#define DPM_PUBLIC DPM_API_EXPORT
#define DPM_PRIVATE DPM_API_HIDDEN
#endif

#ifdef DPM_INLINE_EXTENSIONS
#define DPM_DECLARE_EXT_NAMESPACE inline namespace ext
#else
#define DPM_DECLARE_EXT_NAMESPACE namespace ext
#endif
