// Swift port of mamba-metal's selective_scan_chunked: arbitrary-seqlen Mamba scan.
// Kernel source mirrors mamba_metal/kernels/selective_scan_chunked.metal verbatim.

import Foundation
import MLX
import MLXFast

let selectiveScanSource = """
    uint t = thread_position_in_threadgroup.x;
    uint batch_id = threadgroup_position_in_grid.y;
    uint dim_id = threadgroup_position_in_grid.z;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint n_sg = simdgroups_per_threadgroup;

    threadgroup float warp_a[32];
    threadgroup float warp_b[32];
    threadgroup float carry_a[64];
    threadgroup float carry_b[64];

    if (t < dstate) {
        carry_a[t] = 1.0;
        carry_b[t] = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint chunk_size = 1024u;
    uint n_chunks = (seqlen + chunk_size - 1u) / chunk_size;

    float D_val = (use_D != 0u) ? D[dim_id] : 0.0;

    for (uint c = 0; c < n_chunks; ++c) {
        uint global_t = c * chunk_size + t;
        bool in_range = global_t < seqlen;

        uint udx = batch_id * dim * seqlen + dim_id * seqlen + global_t;
        float u_t = in_range ? u[udx] : 0.0;
        float delta_t_raw = in_range ? delta[udx] : 0.0;

        float delta_t;
        if (apply_softplus != 0u) {
            delta_t = (delta_t_raw <= 20.0) ? log(1.0 + exp(delta_t_raw)) : delta_t_raw;
        } else {
            delta_t = delta_t_raw;
        }

        float y_t = in_range ? (D_val * u_t) : 0.0;

        for (uint s = 0; s < dstate; ++s) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float A_ds = A[dim_id * dstate + s];
            uint bcdx = batch_id * dstate * seqlen + s * seqlen + global_t;
            float B_st = in_range ? B[bcdx] : 0.0;
            float C_st = in_range ? C[bcdx] : 0.0;

            float a = in_range ? exp(delta_t * A_ds) : 1.0;
            float b = in_range ? (delta_t * u_t * B_st) : 0.0;

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
                float ca_intra = warp_a[sg - 1u];
                float cb_intra = warp_b[sg - 1u];
                b = a * cb_intra + b;
                a = a * ca_intra;
            }

            float ca = carry_a[s];
            float cb = carry_b[s];
            b = a * cb + b;

            if (in_range) {
                y_t += b * C_st;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (t == 0u) {
                float block_a = warp_a[n_sg - 1u];
                float block_b = warp_b[n_sg - 1u];
                carry_a[s] = block_a * ca;
                carry_b[s] = block_a * cb + block_b;
            }
        }

        if (use_z != 0u && in_range) {
            float z_val = z[udx];
            y_t = y_t * (z_val / (1.0 + exp(-z_val)));
        }

        if (in_range) {
            y[batch_id * dim * seqlen + dim_id * seqlen + global_t] = y_t;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < dstate) {
        ssm_state_out[(batch_id * dim + dim_id) * dstate + t] = carry_b[t];
    }
"""

let selectiveScanKernel = MLXFast.metalKernel(
    name: "selective_scan_chunked",
    inputNames: [
        "u", "delta", "A", "B", "C", "D", "z",
        "batch", "dim", "dstate", "seqlen",
        "apply_softplus", "use_D", "use_z"
    ],
    outputNames: ["y", "ssm_state_out"],
    source: selectiveScanSource
)

// Swift wrapper mirroring Python's selective_scan(...) API.
// Returns (y, ssm_state). D / z / softplus default to off.
func selectiveScan(
    u: MLXArray, delta: MLXArray, A: MLXArray,
    B: MLXArray, C: MLXArray,
    D: MLXArray? = nil, z: MLXArray? = nil,
    deltaSoftplus: Bool = false
) -> (MLXArray, MLXArray) {
    let batch = u.shape[0]
    let dim = u.shape[1]
    let seqlen = u.shape[2]
    let dstate = A.shape[1]

    let useD = D != nil
    let useZ = z != nil
    let dArg = D ?? MLX.zeros([dim], type: Float32.self)
    let zArg = z ?? MLX.zeros([1], type: Float32.self)

    let outputs = selectiveScanKernel(
        [
            u, delta, A, B, C, dArg, zArg,
            MLXArray(UInt32(batch)),
            MLXArray(UInt32(dim)),
            MLXArray(UInt32(dstate)),
            MLXArray(UInt32(seqlen)),
            MLXArray(UInt32(deltaSoftplus ? 1 : 0)),
            MLXArray(UInt32(useD ? 1 : 0)),
            MLXArray(UInt32(useZ ? 1 : 0))
        ],
        grid: (1024, batch, dim),
        threadGroup: (1024, 1, 1),
        outputShapes: [u.shape, [batch, dim, dstate]],
        outputDTypes: [u.dtype, .float32]
    )
    return (outputs[0], outputs[1])
}

// CPU reference (sequential recurrence)
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

// --- Run a test ---
let batch = 1, dim = 2, dstate = 8, seqlen = 256

// Reproducible pseudo-random inputs (simple LCG so Python and Swift can match seeds if needed).
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

let uArr = MLXArray(u, [batch, dim, seqlen])
let deltaArr = MLXArray(delta, [batch, dim, seqlen])
let AArr = MLXArray(A, [dim, dstate])
let BArr = MLXArray(B, [batch, dstate, seqlen])
let CArr = MLXArray(C, [batch, dstate, seqlen])

let (yMx, _) = selectiveScan(u: uArr, delta: deltaArr, A: AArr, B: BArr, C: CArr)
eval(yMx)

let yRef = referenceSelectiveScan(
    u: u, delta: delta, A: A, B: B, C: C,
    batch: batch, dim: dim, dstate: dstate, seqlen: seqlen
)
let yMetal = yMx.asArray(Float.self)

var maxErr: Float = 0
for i in 0..<yRef.count {
    maxErr = max(maxErr, abs(yMetal[i] - yRef[i]))
}

print("B=\(batch) D=\(dim) N=\(dstate) T=\(seqlen)")
print("metal y[0..4]   = \(Array(yMetal.prefix(5)))")
print("ref   y[0..4]   = \(Array(yRef.prefix(5)))")
print("metal y[-4..]   = \(Array(yMetal.suffix(4)))")
print("ref   y[-4..]   = \(Array(yRef.suffix(4)))")
print("max abs err = \(maxErr)")
assert(maxErr < 1e-4, "selective_scan diverged from reference")
print("OK — selective_scan_chunked Swift matches CPU reference.")
