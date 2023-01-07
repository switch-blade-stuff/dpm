/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#include "arch.hpp"

#if !defined(NDEBUG) && !defined(SVM_DEBUG)
#define SVM_DEBUG
#endif

/* Define SVM_USE_IMPORT only if modules are enabled and supported by the compiler. */
#if defined(SVM_USE_MODULES) && defined(__cpp_modules)
#define SVM_USE_IMPORT
#endif

#ifdef SVM_USE_IMPORT

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
#define SVM_ASSUME(x)
#elif defined(_MSC_VER)
#define SVM_ASSUME(x) __assume(x)
#elif 0 && defined(__clang__) /* See https://github.com/llvm/llvm-project/issues/55636 and https://github.com/llvm/llvm-project/issues/45902 */
#define SVM_ASSUME(x) __builtin_assume(x)
#elif defined(__GNUC__)
#define SVM_ASSUME(x) if (!(x)) __builtin_unreachable()
#else
#define SVM_ASSUME(x)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define SVM_UNREACHABLE() __builtin_unreachable()
#elif defined(__cpp_lib_unreachable) && __cpp_lib_unreachable >= 202202L
#define SVM_UNREACHABLE() std::unreachable()
#else
#define SVM_UNREACHABLE() SVM_ASSUME(false)
#endif

#if defined(_MSC_VER)

#define SVM_PURE
#define SVM_TARGET(t)
#define SVM_FORCEINLINE __forceinline
#define SVM_NEVER_INLINE __declspec(noinline)

/* MSVC inserts security cookies into fixed-size arrays even if the index is known at compile-time. Stop it. */
#define SVM_SAFE_ARRAY __declspec(safebuffers)
/* Windows calling convention will never use vector registers for function arguments. Force it. */
#define SVM_VECTORCALL __vectorcall

#elif defined(__clang__) || defined(__GNUC__)

#define SVM_PURE __attribute__((pure))
#define SVM_TARGET(t) __attribute__((target(t)))
#define SVM_FORCEINLINE __attribute__((always_inline))
#define SVM_NEVER_INLINE __attribute__((noinline))
#define SVM_SAFE_ARRAY
#define SVM_VECTORCALL

#else

#define SVM_PURE
#define SVM_TARGET(t)
#define SVM_FORCEINLINE
#define SVM_NEVER_INLINE
#define SVM_SAFE_ARRAY
#define SVM_VECTORCALL

#endif

#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)

#define SVM_API_HIDDEN
#if defined(_MSC_VER)
#define SVM_API_EXPORT __declspec(dllexport)
#define SVM_API_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#define SVM_API_EXPORT __attribute__((dllexport))
#define SVM_API_IMPORT __attribute__((dllimport))
#endif

#elif __GNUC__ >= 4

#define SVM_API_HIDDEN __attribute__((visibility("hidden")))
#define SVM_API_EXPORT __attribute__((visibility("default")))
#define SVM_API_IMPORT __attribute__((visibility("default")))

#else

#define SVM_API_HIDDEN
#define SVM_API_EXPORT
#define SVM_API_IMPORT

#endif

#ifdef SVM_EXPORT
#define SVM_PUBLIC SVM_API_IMPORT
#define SVM_PRIVATE SVM_API_HIDDEN
#else
#define SVM_PUBLIC SVM_API_EXPORT
#define SVM_PRIVATE SVM_API_HIDDEN
#endif

#ifdef SVM_INLINE_EXTENSIONS
#define SVM_DECLARE_EXT_NAMESPACE inline namespace ext
#else
#define SVM_DECLARE_EXT_NAMESPACE namespace ext
#endif