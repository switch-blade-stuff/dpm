/*
 * Created by switchblade on 2022-12-31.
 */

#pragma once

#if defined(i386) || defined(__i386__) || defined(__i386) || defined(__x86_64__) || defined(_M_X86) || defined(_M_IX86) || defined(_M_X64)
#define DPM_ARCH_X86

#if defined(__SSE__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 1) || defined(_M_AMD64) || defined(_M_X64)
#define DPM_HAS_SSE
#endif

#if defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_AMD64) || defined(_M_X64)
#define DPM_HAS_SSE2
#endif

#ifdef __SSE3__
#define DPM_HAS_SSE3
#endif

#ifdef __SSSE3__
#define DPM_HAS_SSSE3
#endif

#ifdef __SSE4_1__
#define DPM_HAS_SSE4_1
#endif
#ifdef __SSE4_2__
#define DPM_HAS_SSE4_2
#endif

#ifdef __AVX__
#define DPM_HAS_AVX

/* MSVC does not define SSE3+ macros, so we need to emulate them. AVX CPUs should support all other SSE levels. */
#if defined(_MSC_VER)
#define DPM_HAS_SSE3
#define DPM_HAS_SSSE3
#define DPM_HAS_SSE4_1
#define DPM_HAS_SSE4_2
#endif
#endif

#ifdef __FMA__
#define DPM_HAS_FMA
#endif

#ifdef __AVX2__
#define DPM_HAS_AVX2

/* If __FMA__ is not defined by the compiler but AVX2 is supported, enable FMA as well. */
#ifndef DPM_HAS_FMA
#define DPM_HAS_FMA
#endif
#endif

#ifdef __AVX512F__
#define DPM_HAS_AVX512
#define DPM_HAS_AVX512F

#ifdef __AVX512CD__
#define DPM_HAS_AVX512CD
#endif
#ifdef __AVX512BW__
#define DPM_HAS_AVX512BW
#endif
#ifdef __AVX512DQ__
#define DPM_HAS_AVX512DQ
#endif
#ifdef __AVX512VL__
#define DPM_HAS_AVX512VL /* AVX operations on non-AVX registers. */
#endif
#endif

#elif defined(__arm__) || defined(__arm) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#define DPM_ARCH_ARM

#if defined(__aarch64__) || defined(_M_ARM64)
#define DPM_ARCH_ARM64
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define DPM_HAS_NEON
#endif

#endif
