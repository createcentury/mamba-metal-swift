// Smoke test for MambaBlock + MambaModel.

import Foundation
import MLX
import MLXRandom
import MambaMetal

// --- MambaBlock single-layer test ---
print("=== MambaBlock ===")
let block = MambaBlock(dModel: 64, dState: 16, dConv: 4, expand: 2)
let xBlock = MLXRandom.normal([1, 32, 64])
let yBlock = block(xBlock)
eval(yBlock)

let yBArr = yBlock.asArray(Float.self)
print("input  \(xBlock.shape)  ->  output \(yBlock.shape)")
print("dInner=\(block.dInner) dtRank=\(block.dtRank)")
print("range [\(yBArr.min()!), \(yBArr.max()!)]  NaN=\(yBArr.contains { $0.isNaN })")
assert(yBlock.shape == [1, 32, 64])

// --- Full MambaModel test ---
print("\n=== MambaModel (small) ===")
let cfg = MambaConfig(dModel: 64, nLayer: 2, vocabSize: 128, dState: 16, dConv: 4, expand: 2)
let model = MambaModel(cfg)

let inputIds = MLXRandom.randInt(low: 0, high: cfg.vocabSize, [1, 16])
let logits = model(inputIds)
eval(logits)

let logitsArr = logits.asArray(Float.self)
print("input_ids \(inputIds.shape)  ->  logits \(logits.shape)")
print("range [\(logitsArr.min()!), \(logitsArr.max()!)]  NaN=\(logitsArr.contains { $0.isNaN })")
assert(logits.shape == [1, 16, cfg.vocabSize])
assert(!logitsArr.contains { $0.isNaN || $0.isInfinite })

print("\nOK — MambaBlock + MambaModel run end-to-end in Swift.")
