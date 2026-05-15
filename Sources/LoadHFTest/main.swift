// Load state-spaces/mamba-130m-hf weights into a Swift MambaModel
// (paths point at the existing HF cache populated by the Python project).

import Foundation
import MLX
import MambaMetal

let home = FileManager.default.homeDirectoryForCurrentUser.path
let snapshot = "\(home)/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots"

func firstSnapshot() -> String? {
    let url = URL(fileURLWithPath: snapshot)
    guard let dirs = try? FileManager.default.contentsOfDirectory(atPath: url.path) else { return nil }
    return dirs.first.map { "\(snapshot)/\($0)" }
}

guard let dir = firstSnapshot() else {
    fatalError("mamba-130m-hf snapshot not found at \(snapshot)")
}

let safetensorsURL = URL(fileURLWithPath: "\(dir)/model.safetensors")
let configURL = URL(fileURLWithPath: "\(dir)/config.json")

print("Loading from: \(dir)")
let start = Date()
let (model, cfg) = try loadMambaHF(safetensorsURL: safetensorsURL, configURL: configURL)
let loadTime = Date().timeIntervalSince(start)
print("loaded in \(String(format: "%.2f", loadTime))s")
print("config: d_model=\(cfg.dModel) n_layer=\(cfg.nLayer) vocab=\(cfg.vocabSize) dt_rank=\(cfg.dtRank ?? -1)")

// Random short input — just verify forward runs end-to-end.
let inputIds = MLXArray((0..<16).map { Int32(($0 * 7919) % cfg.vocabSize) }, [1, 16])
let t0 = Date()
let logits = model(inputIds)
eval(logits)
let fwdTime = Date().timeIntervalSince(t0)
print("forward: \(String(format: "%.0f ms", fwdTime * 1000))  logits shape \(logits.shape)")

let arr = logits.asArray(Float.self)
let anyNaN = arr.contains { $0.isNaN }
let anyInf = arr.contains { $0.isInfinite }
print("any NaN: \(anyNaN)  any Inf: \(anyInf)")
print("range: [\(String(format: "%.2f", arr.min()!)), \(String(format: "%.2f", arr.max()!))]")

// Argmax per position
let argmax = MLX.argMax(logits, axis: -1)
eval(argmax)
let am = argmax.asArray(Int32.self)
let unique = Set(am)
print("argmax token ids: \(am.prefix(8))…  unique=\(unique.count)/\(am.count)")

assert(!anyNaN && !anyInf)
print("\nOK — mamba-130m-hf loaded from Swift and forward runs.")
