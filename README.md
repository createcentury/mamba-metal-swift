# mamba-metal-swift

Swift port of [mamba-metal](https://github.com/createcentury/mamba-metal) — Mamba's selective scan in Metal Shading Language, running on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).

Currently a minimal smoke test (`PairScanTest`) verifying that the `.metal` kernels from the Python project run identically when JIT-compiled through `MLXFast.metalKernel` in Swift.

## Status

- ✅ `pair_scan` kernel — runs on Mac GPU and **iPhone GPU** (max abs err < 1e-6)
- ✅ `selective_scan_chunked` kernel — full Mamba selective scan (D / softplus / z / state output flags), Mac + **iPhone** (max abs err ≈ 3e-7)
- ✅ `MambaMetal` library — importable from any Swift / SwiftUI iOS app
- ✅ iOS demo app: [createcentury/mamba-metal-demo](https://github.com/createcentury/mamba-metal-demo) — both kernels verified end-to-end on iPhone Air
- 🚧 MambaBlock / MambaModel layer (nn.Linear / Conv1d / RMSNorm) — TODO
- 🚧 HF safetensors loader + GPT-NeoX tokenizer — TODO
- 🚧 Full Mamba inference on iPhone — TODO

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
