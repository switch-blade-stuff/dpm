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
    <td>SVM_DYNAMIC_DISPATCH</td>
    <td>-DSVM_DYNAMIC_DISPATCH</td>
    <td>ON</td>
    <td>Toggles runtime detection & dispatch of SIMD operations</td>
  </tr>
  <tr>
    <td>SVM_USE_AVX512</td>
    <td>-DSVM_USE_AVX512</td>
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
```

Build artifacts will be found in `build/bin` and `build/lib`. Minimum required CMake version is 3.19

In order to use the library as a CMake link dependency, you must link to one of the following targets:

* `svm-interface` - interface headers of the library.
* `svm` - static or shared library target, depending on the value of `BUILD_SHARED_LIBS`.

## Notes

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

In order to take advantage of AVX512 SIMD operations, use the `wide<T>` extension ABI tag.
To enable AVX512 operations globally, use the `SVM_USE_AVX512` option.
