/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#if defined(i386) || defined(__i386__) || defined(__i386) || defined(__x86_64__) || defined(_M_X86) || defined(_M_IX86) || defined(_M_X64)
#define SVM_ARCH_X86

#if defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1) || defined(_M_AMD64) || defined(_M_X64)
#define SVM_HAS_SSE
#endif

#if defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_AMD64) || defined(_M_X64)
#define SVM_HAS_SSE2
#endif

#ifdef __SSE3__
#define SVM_HAS_SSE3
#endif

#ifdef __SSSE3__
#define SVM_HAS_SSSE3
#endif

#ifdef __SSE4_1__
#define SVM_HAS_SSE4_1
#endif
#ifdef __SSE4_2__
#define SVM_HAS_SSE4_2
#endif

#ifdef __AVX__
#define SVM_HAS_AVX

#ifdef __AVX2__
#define SVM_HAS_AVX2
#endif
#endif

/* TODO: Support AVX512 */

#elif defined(__arm__) || defined(__arm) || defined(__aarch64__) || defined(_M_ARM)
#define SVM_ARCH_ARM

#if defined(SVM_ARCH_ARM) && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#define SVM_HAS_NEON
#endif

#endif
