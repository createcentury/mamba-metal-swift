# mamba-metal-swift

Swift port of [mamba-metal](https://github.com/createcentury/mamba-metal) — Mamba's selective scan in Metal Shading Language, running on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift). Same `.metal` kernel sources, JIT-compiled through `MLXFast.metalKernel`.

This repo contains both the library (`MambaMetal`) and a SwiftUI iOS demo (`Demo/`) that loads a real HuggingFace Mamba checkpoint and generates text on the iPhone's GPU.

## Capabilities

- ✅ Selective scan kernel (`pair_scan`, `selective_scan_chunked`) — Mac and **iPhone GPU**
- ✅ Full Swift Mamba stack: `MambaBlock`, `MambaResidualBlock`, `MambaModel`
- ✅ HF safetensors loader + GPT-NeoX tokenizer (via `swift-transformers`)
- ✅ O(1)-per-token incremental decode (`prefill` + `step` state caching)
- ✅ `mamba-130m-hf` end-to-end inference on **iPhone Air**, ~10 ms/token decode

## Library

```swift
import MambaMetal

let (model, _) = try loadMambaHF(safetensorsURL: …, configURL: …)
let tokenizer = try await AutoTokenizer.from(modelFolder: …, strict: false)
let text = greedyGenerateFast(
    model: model, tokenizer: tokenizer,
    prompt: "The capital of Japan is",
    maxNewTokens: 50
)
```

## Demo app (iOS / iPadOS)

```bash
brew install xcodegen
cd Demo
xcodegen generate
xcodebuild -scheme MambaMetalDemo -configuration Release \
  -destination 'platform=iOS,name=<your-iphone-name>' \
  -allowProvisioningUpdates -skipMacroValidation build
```

The app downloads `state-spaces/mamba-130m-hf` on first launch (~520 MB), then exposes:
- Prompt + Generate button (fast / slow decode toggle, max-tokens slider)
- "Auto sweep" that runs a fixed measurement grid and writes JSON to the app's Documents directory (pull with `xcrun devicectl device copy from`)
- Kernel smoke tests (`pair_scan`, `selective_scan_chunked` vs CPU reference)

## CLI tests (macOS)

`PairScanTest`, `SelectiveScanTest`, `MambaBlockTest`, `LoadHFTest`, `GenerateTest` cover progressively the kernel, model, weight loading and generation paths. Build via:

```bash
xcodebuild -scheme GenerateTest -configuration Release \
  -destination 'platform=macOS' -skipMacroValidation build
```

## Why a separate package

`mlx-swift`'s LLM registry currently has no Mamba implementation. This package fills that gap on the Swift / iOS side, and pairs with [mamba-metal](https://github.com/createcentury/mamba-metal) (Python) and [mamba-vs-transformer-edge](https://github.com/createcentury/mamba-vs-transformer-edge) (benchmark study).

## License

MIT
