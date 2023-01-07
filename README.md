# Standard Vector Math

This library contains a feature-complete implementation of the C++ Data-parallel vector library TS with optional
extensions.

## Compiler support

## SIMD intrinsics support

The library currently supports the following architectures for SIMD operations:

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
- ARM Neon

On architectures without SIMD intrinsic support, SIMD operations are emulated via single-data operations.

## Library options

<table>
  <tr><th>#define macro</th><th>CMake option</th><th>Default value</th><th>Description</th></tr>
  <tr>
    <td>SVM_USE_MODULES</td>
    <td>-DSVM_USE_MODULES</td>
    <td>ON</td>
    <td>Toggles support for C++20 modules</td>
  </tr>
  <tr>
    <td>SVM_INLINE_EXTENSIONS</td>
    <td>-DSVM_INLINE_EXTENSIONS</td>
    <td>ON</td>
    <td>Toggles inlining of the library extension namespace (see notes)</td>
  </tr>
  <tr>
    <td>SVM_DYNAMIC_DISPATCH</td>
    <td>-DSVM_DYNAMIC_DISPATCH</td>
    <td>ON</td>
    <td>Toggles runtime detection & dispatch of SIMD operations</td>
  </tr>
  <tr>
    <td>SVM_NATIVE_AVX512</td>
    <td>-DSVM_NATIVE_AVX512</td>
    <td>OFF</td>
    <td>Toggles global support for AVX512 instructions on x86 targets (see notes)</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>-DSVM_USE_IPO</td>
    <td>ON</td>
    <td>Toggles support for inter-procedural optimization</td>
  </tr>
  <tr>
    <td>N/A</td>
    <td>-DSVM_TESTS</td>
    <td>OFF</td>
    <td>Enables unit test target</td>
  </tr>
</table>

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

* `svm-interface` - interface headers of the library.
* `svm` - static or shared library target, depending on the value of `BUILD_SHARED_LIBS`.

## Notes

### Extensions

The library provides the following extensions to the standard API:

* ABI tags
    * x86
        * `sse`
        * `avx`
        * `avx512`
    * ARM
        * `neon`
* Storage traits & accessors
  * `native_data_type`
  * `native_data_size`
  * `std::span to_native_data(simd &)`
  * `std::span to_native_data(const simd &)`
  * `std::span to_native_data(simd_mask &)`
  * `std::span to_native_data(const simd_mask &)`
* Blend functions
    * `simd blend(const simd &, const simd &, const simd_mask &)`
    * `simd blend(const simd &, const const_where_expression &)`
    * `simd_mask blend(const simd_mask &, const simd_mask &, const simd_mask &)`
    * `simd_mask blend(const simd_mask &, const const_where_expression &)`
    * `T blend(const T &, const T &, /* bool-wrapper */)`
    * `T blend(const T &, const const_where_expression &)`

All extensions are available from the `svm::ext` and `svm::simd_abi::ext` namespaces. If `SVM_INLINE_EXTENSIONS` is
defined, the `ext` namespaces are declared as inline.

### Dynamic dispatch

By default, SVM will attempt to automatically detect SIMD support and dispatch vectorized functions appropriately on
supported CPUs. This enables the library to support multiple tiers of hardware with the same binary. This, however, may
increase binary size and add additional overhead to every dispatched operation. In order to disable dynamic dispatch,
use the `SVM_DYNAMIC_DISPATCH` option.

### AVX512

By default, the maximum SIMD level used by the library on x86 platforms is FMA. AVX512 is intentionally ignored, due to
inefficiency of AVX512 operations on certain CPUs. See the following articles for details:

- [https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/](https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/)
- [https://news.ycombinator.com/item?id=21031905](https://news.ycombinator.com/item?id=21031905)
- [https://www.phoronix.com/news/Linus-Torvalds-On-AVX-512](https://www.phoronix.com/news/Linus-Torvalds-On-AVX-512)

In order to take advantage of AVX512 SIMD operations, use the `avx512<T>` extension ABI tag.
To enable AVX512 operations globally, use the `SVM_NATIVE_AVX512` option.
