/*
 * Created by switchblade on 2023-01-14.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "../define.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "mbase.hpp"
#include "class.hpp"
#include "sign.hpp"
#include "fmadd.hpp"
#include "sincos.hpp"

#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif