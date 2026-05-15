# mamba-metal-swift

Swift port of [mamba-metal](https://github.com/createcentury/mamba-metal) — Mamba's selective scan in Metal Shading Language, running on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).

Currently a minimal smoke test (`PairScanTest`) verifying that the `.metal` kernels from the Python project run identically when JIT-compiled through `MLXFast.metalKernel` in Swift.

## Status

- ✅ `pair_scan` kernel runs from Swift, output matches CPU reference exactly (max abs err = 0).
- 🚧 selective_scan_chunked port — TODO
- 🚧 MambaBlock / MambaModel in Swift — TODO
- 🚧 HF safetensors loader — TODO
- 🚧 iOS app integration — TODO

## Build

Requires Xcode 26+, Metal Toolchain installed (`xcodebuild -downloadComponent MetalToolchain`).

```bash
xcodebuild -scheme mamba-metal-swift -configuration Release \
  -destination 'platform=macOS' -skipMacroValidation build
```

Run:

```bash
EXEC=$(find ~/Library/Developer/Xcode/DerivedData/mamba-metal-swift-*/Build/Products/Release/PairScanTest)
DYLD_FRAMEWORK_PATH=$(dirname "$EXEC") "$EXEC"
```

## Why a separate package

`mlx-swift`'s LLM registry currently has no Mamba implementation. This package is the Swift counterpart to the Python [mamba-metal](https://github.com/createcentury/mamba-metal), targeting the same Metal kernels but consumable from Swift / iOS apps.

## License

MIT
