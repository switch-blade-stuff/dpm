/*
 * Created by switch_blade on 2023-02-10.
 */

#pragma once

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
#define DPM_API_PUBLIC DPM_API_EXPORT
#define DPM_API_PRIVATE DPM_API_HIDDEN
#else
#define DPM_API_PUBLIC DPM_API_IMPORT
#define DPM_API_PRIVATE DPM_API_HIDDEN
#endif
