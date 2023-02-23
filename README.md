# Data-Parallel Math

This library implements a feature-complete version of the C++ data-parallel math library TS with optional extensions.

## SIMD intrinsics support

DPM currently supports the following architectures for SIMD vectorization:

- x86
    - SSE
    - SSE2
    - SSE3
    - SSSE3
    - SSE4.1
    - SSE4.2
    - AVX
    - AVX2
    - FMA
    - AVX512 (see notes)

On architectures without SIMD support, vectorization is emulated via scalar operations.

## Library options

<table>
  <tr><th>#define macro</th><th>CMake option</th><th>Default value</th><th>Description</th></tr>
  <tr>
    <td>DPM_INLINE_EXTENSIONS</td>
    <td>-DDPM_INLINE_EXTENSIONS</td>
    <td>ON</td>
    <td>Toggles inlining of the library extension namespace (see notes)</td>
  </tr>
  <tr>
    <td>DPM_HANDLE_ERRORS</td>
    <td>-DDPM_HANDLE_ERRORS</td>
    <td>ON</td>
    <td>Toggles detection of math errors & reporting via <a hred="https://en.cppreference.com/w/cpp/numeric/math/math_errhandling">math_errhandling</a> (see notes)</td>
  </tr>
  <tr>
    <td>DPM_PROPAGATE_NAN</td>
    <td>-DDPM_PROPAGATE_NAN</td>
    <td>ON</td>
    <td>Toggles guaranteed propagation of NaN (see notes)</td>
  </tr>
  <tr>
    <td>DPM_USE_SVML</td>
    <td>-DDPM_USE_SVML</td>
    <td>OFF</td>
    <td>Enables use of math functions provided by SVML (see notes)</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>-DDPM_USE_IPO</td>
    <td>ON</td>
    <td>Toggles support for inter-procedural optimization</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>-DDPM_TESTS</td>
    <td>OFF</td>
    <td>Enables unit test target</td>
  </tr>
</table>

Note that the default value applies only to CMake options.

## Build & usage instructions

In order to build the library using CMake, run the following commands:

```shell
mkdir build
cd build
cmake ..
cmake --build .
```

Build artifacts will be found in `build/bin` and `build/lib`. Minimum required CMake version is 3.20

In order to use the library as a CMake link dependency, you must link to one of the following targets:

* `dpm-interface` - interface headers of the library.
* `dpm` - static or shared library target, depending on the value of `BUILD_SHARED_LIBS`.

## Notes

### Extensions

DPM provides the following utilities and extensions to the standard API:

* ABI tags
    * `struct aligned_vector`
    * `using packed_buffer = implementation-defined`
    * `using common = implementation-defined`
    * x86
        * `using sse = implementation-defined`
        * `using avx = implementation-defined`
* Storage traits & accessors
    * `struct native_data_type`
    * `struct native_data_size`
    * `std::span to_native_data(simd<T, Abi> &)`
    * `std::span to_native_data(const simd<T, Abi> &)`
    * `std::span to_native_data(simd_mask<T, Abi> &)`
    * `std::span to_native_data(const simd_mask<T, Abi> &)`
* Blend functions
    * `simd<T, Abi> blend(const simd<T, Abi> &, const simd<T, Abi> &, const simd_mask<T, Abi> &)`
    * `simd<T, Abi> blend(const simd<T, Abi> &, const const_where_expression &)`
    * `simd_mask<T, Abi> blend(const simd_mask<T, Abi> &, const simd_mask<T, Abi> &, const simd_mask<T, Abi> &)`
    * `simd_mask<T, Abi> blend(const simd_mask<T, Abi> &, const const_where_expression &)`
    * `V blend(const V &, const V &, /* bool-wrapper */)`
    * `V blend(const V &, const const_where_expression &)`
* Shuffle functions
    * `simd<T, Abi> shuffle<Is...>(const simd<T, Abi> &)`
    * `simd_mask<T, Abi> shuffle<Is...>(const simd_mask<T, Abi> &)`
    * `simd_mask<T, Abi> shuffle<Is...>(const V &)`
* Constant bit shifts
    * `simd<T, Abi> lsl<N>(const simd<T, Abi> &)`
    * `simd<T, Abi> lsr<N>(const simd<T, Abi> &)`
    * `simd<T, Abi> asl<N>(const simd<T, Abi> &)`
    * `simd<T, Abi> asr<N>(const simd<T, Abi> &)`
* Reductions
    * `T hadd(const simd<T, Abi> &)`
    * `T hmul(const simd<T, Abi> &)`
    * `T hand(const simd<T, Abi> &)`
    * `T hxor(const simd<T, Abi> &)`
    * `T hor(const simd<T, Abi> &)`
* Basic math functions
    * `simd<T, Abi> remquo(const simd<T, Abi> &, const simd<T, Abi> &, simd<T, Abi> &)`
    * `simd<T, Abi> nan<T, Abi>(const char *)`
* Power math functions
    * `simd<T, Abi> rcp(const simd<T, Abi> &)`
    * `simd<T, Abi> rsqrt(const simd<T, Abi> &)`
* Nearest integer functions
    * `rebind_simd_t<I, simd<T, Abi>> iround<I>(const simd<T, Abi> &)`
    * `rebind_simd_t<I, simd<T, Abi>> irint<I>(const simd<T, Abi> &)`
    * `rebind_simd_t<I, simd<T, Abi>> itrunc<I>(const simd<T, Abi> &)`
    * `rebind_simd_t<long, simd<T, Abi>> ltrunc(const simd<T, Abi> &)`
    * `rebind_simd_t<long long, simd<T, Abi>> lltrunc(const simd<T, Abi> &)`
* Floating-point manipulation functions
    * `simd<T, Abi> frexp(const simd<T, Abi> &x, simd<int, Abi> &)`
    * `simd<T, Abi> modf(const simd<T, Abi> &x, simd<T, Abi> &)`
* Optimization hints
    * `#define DPM_UNREACHABLE()`
    * `#define DPM_NEVER_INLINE`
    * `#define DPM_FORCEINLINE`
    * `#define DPM_ASSUME(cnd)`
* Assertions & traps
    * `#define DPM_ASSERT_MSG_LOC_ALWAYS(cnd, msg, src_loc)`
    * `#define DPM_ASSERT_MSG_LOC(cnd, msg, src_loc)`
    * `#define DPM_ASSERT_MSG_ALWAYS(cnd, msg)`
    * `#define DPM_ASSERT_MSG(cnd, msg)`
    * `#define DPM_ASSERT_ALWAYS(cnd)`
    * `#define DPM_ASSERT(cnd)`
    * `#define DPM_DEBUGTRAP()`
* Other utilities
    * `class cpuid`

Additionally, versions of some operators and math functions accepting a scalar as one of the arguments are provided.

All extensions are available from the `dpm::ext` and `dpm::simd_abi::ext` namespaces. If `DPM_INLINE_EXTENSIONS` option
is enabled, the `ext` namespaces are declared as inline.

### Error handling & NaN

The standard specifies that floating-point math functions such as `sin`, `cos`, etc. must report math errors via the
mechanism specified in [math_errhandling](https://en.cppreference.com/w/cpp/numeric/math/math_errhandling).
When `DPM_HANDLE_ERRORS` option is enabled, DPM catches and reports errors as specified by the standard, however this
does reduce efficiency of math functions due to the additional domain checks. If performance is preferred over
accuracy, disable `DPM_HANDLE_ERRORS`. Note that if error handling is disabled, domain, over- and underflow checks will
not be preformed.

Additionally, if `DPM_PROPAGATE_NAN` option is enabled, the library guarantees that math functions will propagate any
NaN inputs, which may be used as a form of error handling. If both `DPM_PROPAGATE_NAN` and `DPM_HANDLE_ERRORS` are
disabled, invoking math functions with `NaN` as input will result in undefined behavior unless otherwise specified.
`DPM_HANDLE_ERRORS` implies `DPM_PROPAGATE_NAN`.

### AVX512

While DPM does utilize AVX512 instructions for 128- and 256-bit operations, there is no support for 512-wide vector data
types. The main reasons being the increased complexity of implementation due to both the fracturing of AVX512 standard,
and complexity of most 512-bit wide instructions (ex. there is no 512 blend, and it must be emulated via more complex
fused operations); as well as relative inefficiency of 512-bit wide registers (for general-purpose use cases) on certain
CPUs. See the following articles for details:

- [https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/](https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/)
- [https://news.ycombinator.com/item?id=21031905](https://news.ycombinator.com/item?id=21031905)
- [https://www.phoronix.com/news/Linus-Torvalds-On-AVX-512](https://www.phoronix.com/news/Linus-Torvalds-On-AVX-512)

### SVML

When `DPM_USE_SVML` is enabled, DPM will use mathematical functions provided by SVML for trigonometric, hyperbolic,
exponential, nearest integer and error functions instead of the built-in implementation. Inlined functions will still
use the internal
implementation. Note that if `DPM_USE_SVML` is enabled, NaN propagation and error handling options are ignored for
affected functions, any error handling is left to SVML.
