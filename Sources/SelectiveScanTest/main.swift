// Smoke test: selective_scan_chunked via the MambaMetal library.

import Foundation
import MLX
import MambaMetal

func referenceSelectiveScan(
    u: [Float], delta: [Float], A: [Float],
    B: [Float], C: [Float],
    batch: Int, dim: Int, dstate: Int, seqlen: Int
) -> [Float] {
    var y = [Float](repeating: 0, count: batch * dim * seqlen)
    for bi in 0..<batch {
        for di in 0..<dim {
            var h = [Float](repeating: 0, count: dstate)
            for t in 0..<seqlen {
                let dt = delta[bi*dim*seqlen + di*seqlen + t]
                let ut = u[bi*dim*seqlen + di*seqlen + t]
                var ySum: Float = 0
                for s in 0..<dstate {
                    let Ads = A[di*dstate + s]
                    let Bst = B[bi*dstate*seqlen + s*seqlen + t]
                    let Cst = C[bi*dstate*seqlen + s*seqlen + t]
                    let at = exp(dt * Ads)
                    let bt = dt * ut * Bst
                    h[s] = at * h[s] + bt
                    ySum += h[s] * Cst
                }
                y[bi*dim*seqlen + di*seqlen + t] = ySum
            }
        }
    }
    return y
}

let batch = 1, dim = 2, dstate = 8, seqlen = 256

func lcg(seed: inout UInt32) -> Float {
    seed = seed &* 1664525 &+ 1013904223
    return Float(seed) / Float(UInt32.max) * 2.0 - 1.0
}

var seed: UInt32 = 12345
let u = (0..<(batch*dim*seqlen)).map { _ in lcg(seed: &seed) }
let delta = (0..<(batch*dim*seqlen)).map { _ in 0.01 + 0.09 * (lcg(seed: &seed) + 1.0) / 2.0 }
let A = (0..<(dim*dstate)).map { _ in -(0.1 + 1.9 * (lcg(seed: &seed) + 1.0) / 2.0) }
let B = (0..<(batch*dstate*seqlen)).map { _ in lcg(seed: &seed) }
let C = (0..<(batch*dstate*seqlen)).map { _ in lcg(seed: &seed) }

let (yMx, _) = selectiveScan(
    u: MLXArray(u, [batch, dim, seqlen]),
    delta: MLXArray(delta, [batch, dim, seqlen]),
    A: MLXArray(A, [dim, dstate]),
    B: MLXArray(B, [batch, dstate, seqlen]),
    C: MLXArray(C, [batch, dstate, seqlen])
)
eval(yMx)

let yRef = referenceSelectiveScan(
    u: u, delta: delta, A: A, B: B, C: C,
    batch: batch, dim: dim, dstate: dstate, seqlen: seqlen
)
let yMetal = yMx.asArray(Float.self)

var maxErr: Float = 0
for i in 0..<yRef.count { maxErr = max(maxErr, abs(yMetal[i] - yRef[i])) }

print("B=\(batch) D=\(dim) N=\(dstate) T=\(seqlen)")
print("max abs err = \(maxErr)")
assert(maxErr < 1e-4)
print("OK — selective_scan_chunked Swift (library) matches CPU reference.")
