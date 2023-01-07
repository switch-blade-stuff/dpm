/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

/* GCC does not like using vector types as template parameters, which is used for `to_native_data`. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "sse/type.hpp"
#include "avx/type.hpp"
#include "avx512/type.hpp"

#pragma GCC diagnostic pop