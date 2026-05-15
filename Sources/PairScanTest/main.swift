// Smoke test: run the pair_scan Metal kernel via MLX-Swift.
// Same kernel body as ~/mamba-metal/mamba_metal/kernels/pair_scan.metal.

import Foundation
import MLX
import MLXFast
import MLXRandom

let pairScanSource = """
    uint i = thread_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint n_sg = simdgroups_per_threadgroup;

    threadgroup float warp_a[32];
    threadgroup float warp_b[32];

    float a = (i < n) ? a_in[i] : 1.0;
    float b = (i < n) ? b_in[i] : 0.0;

    for (uint d = 1u; d < 32u; d <<= 1) {
        float a_prev = simd_shuffle_up(a, d);
        float b_prev = simd_shuffle_up(b, d);
        if (lane >= d) {
            b = a * b_prev + b;
            a = a * a_prev;
        }
    }

    if (lane == 31u) {
        warp_a[sg] = a;
        warp_b[sg] = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0u) {
        float ta = (lane < n_sg) ? warp_a[lane] : 1.0;
        float tb = (lane < n_sg) ? warp_b[lane] : 0.0;
        for (uint d = 1u; d < 32u; d <<= 1) {
            float ta_prev = simd_shuffle_up(ta, d);
            float tb_prev = simd_shuffle_up(tb, d);
            if (lane >= d) {
                tb = ta * tb_prev + tb;
                ta = ta * ta_prev;
            }
        }
        if (lane < n_sg) {
            warp_a[lane] = ta;
            warp_b[lane] = tb;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg > 0u) {
        float ca = warp_a[sg - 1u];
        float cb = warp_b[sg - 1u];
        b = a * cb + b;
        a = a * ca;
    }

    if (i < n) {
        a_out[i] = a;
        h_out[i] = b;
    }
"""

let kernel = MLXFast.metalKernel(
    name: "pair_scan",
    inputNames: ["a_in", "b_in", "n"],
    outputNames: ["a_out", "h_out"],
    source: pairScanSource
)

// CPU reference: sequentially compute h_i = a_i * h_{i-1} + b_i.
func referenceScan(a: [Float], b: [Float]) -> [Float] {
    var h: Float = 0
    var out = [Float](repeating: 0, count: a.count)
    for i in 0..<a.count {
        h = a[i] * h + b[i]
        out[i] = h
    }
    return out
}

// Test: constant a=0.5, b=1.0 → h_∞ = 2
let n = 1024
let aHost = [Float](repeating: 0.5, count: n)
let bHost = [Float](repeating: 1.0, count: n)

let aArr = MLXArray(aHost, [n])
let bArr = MLXArray(bHost, [n])
let nArr = MLXArray(UInt32(n))

let outputs = kernel(
    [aArr, bArr, nArr],
    grid: (1024, 1, 1),
    threadGroup: (1024, 1, 1),
    outputShapes: [[n], [n]],
    outputDTypes: [.float32, .float32]
)

let hOut = outputs[1]
eval(hOut)

let ref = referenceScan(a: aHost, b: bHost)
let metalArr = hOut.asArray(Float.self)

var maxErr: Float = 0
for i in 0..<n {
    maxErr = max(maxErr, abs(metalArr[i] - ref[i]))
}

print("n = \(n), constant a=0.5, b=1 (expect h→2)")
print("metal h[0..4]   = \(Array(metalArr.prefix(5)))")
print("metal h[1020..] = \(Array(metalArr.suffix(4)))")
print("ref   h[1020..] = \(Array(ref.suffix(4)))")
print("max abs err = \(maxErr)")
assert(maxErr < 1e-4, "pair_scan diverged from reference")
print("OK — pair_scan Metal kernel runs from Swift, matches CPU reference.")
