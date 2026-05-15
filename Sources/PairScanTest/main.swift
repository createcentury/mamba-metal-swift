// Smoke test: pair_scan via the MambaMetal library.

import Foundation
import MLX
import MambaMetal

func referenceScan(a: [Float], b: [Float]) -> [Float] {
    var h: Float = 0
    var out = [Float](repeating: 0, count: a.count)
    for i in 0..<a.count {
        h = a[i] * h + b[i]
        out[i] = h
    }
    return out
}

let n = 1024
let aHost = [Float](repeating: 0.5, count: n)
let bHost = [Float](repeating: 1.0, count: n)

let (_, hOut) = pairScan(a: MLXArray(aHost, [n]), b: MLXArray(bHost, [n]))
eval(hOut)

let ref = referenceScan(a: aHost, b: bHost)
let metalArr = hOut.asArray(Float.self)

var maxErr: Float = 0
for i in 0..<n { maxErr = max(maxErr, abs(metalArr[i] - ref[i])) }

print("n = \(n), constant a=0.5, b=1 (expect h→2)")
print("metal h[1020..] = \(Array(metalArr.suffix(4)))")
print("max abs err = \(maxErr)")
assert(maxErr < 1e-4)
print("OK")
