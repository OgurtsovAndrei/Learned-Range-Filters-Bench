# Vendored Dependencies: LeMonHash

To ensure project stability, reproducibility for research, and cross-platform compatibility, the LeMonHash library and its dependencies have been "vendored" (copied directly into the repository).

## 1. Vendored Libraries
The following libraries are included in `mmph/lemonhash/ext/LeMonHash`:
- **LeMonHash:** Core Learned Monotone Minimal Perfect Hashing implementation.
- **PGM-index:** Piecewise Geometric Model index.
- **sdsl-lite:** Succinct Data Structure Library.
- **SimpleRibbon (BuRR):** Bucketed Ribbon Retrieval for error correction.
- **tlx:** C++ helper library.
- **ips2ra:** In-place Parallel Super Scalar Radix Sort.

## 2. Rationale
- **Stability:** Prevents "broken builds" if external repositories are deleted or moved.
- **Cross-Platform Compatibility:** The original code contained system-specific headers (e.g., `<bits/stdint-uintn.h>`) and assumptions about x86 architectures that caused build failures on Apple Silicon (ARM64). We applied custom patches to make the library standard-compliant.
- **Reproducibility:** Ensures that the exact version of the algorithm used in the thesis is preserved.

## 3. Applied Patches (Highlights)
- Replaced non-standard `<bits/stdint-uintn.h>` with standard `<stdint.h>`.
- Updated `std::result_of` (deprecated in C++17, removed in C++20) to `std::invoke_result_t` for C++20 compatibility.
- Added missing `<sstream>` and `<ostream>` includes.
- Fixed `std::min` type deduction issues.
- Disabled x86-specific intrinsics (`-DSUCCINCT_USE_INTRINSICS=OFF`) to support ARM64/Apple Silicon.

## 4. Cross-Platform Note
The patches applied are standard-compliant. This means the project remains fully compatible with **Linux x86_64** while now supporting **macOS ARM64**. Performance remains high on all platforms.

## 5. Licenses
All original license files (`LICENSE`, `COPYING`) have been preserved in the respective subdirectories. These libraries are primarily licensed under **Apache 2.0**, **MIT**, or **BSD**, which permit redistribution and modification.
