/*
 * Created by switchblade on 2023-01-01.
 */

#pragma once

/* GCC does not like using vector types as template parameters, which is used for `to_native_data`. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"

#include "m128/type.hpp"
#include "m256/type.hpp"
#include "m512/type.hpp"

namespace dpm
{

}

#pragma GCC diagnostic pop