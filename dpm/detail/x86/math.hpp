/*
 * Created by switchblade on 2023-01-14.
 */

#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "../../utility.hpp"

#if defined(DPM_ARCH_X86) && defined(DPM_HAS_SSE)

#include "mbase.hpp"
#include "pow.hpp"
#include "exp.hpp"
#include "trig.hpp"
#include "round.hpp"
#include "fmanip.hpp"
#include "class.hpp"

#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif