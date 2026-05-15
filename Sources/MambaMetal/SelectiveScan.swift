// Swift entry points for the Metal kernels — usable from macOS and iOS apps.

import Foundation
import MLX
import MLXFast

public enum MambaMetalKernels {

    public static let pairScan = MLXFast.metalKernel(
        name: "pair_scan",
        inputNames: ["a_in", "b_in", "n"],
        outputNames: ["a_out", "h_out"],
        source: Kernels.pairScan
    )

    public static let selectiveScanChunked = MLXFast.metalKernel(
        name: "selective_scan_chunked",
        inputNames: [
            "u", "delta", "A", "B", "C", "D", "z",
            "batch", "dim", "dstate", "seqlen",
            "apply_softplus", "use_D", "use_z"
        ],
        outputNames: ["y", "ssm_state_out"],
        source: Kernels.selectiveScanChunked
    )
}

/// Run a (a, b) pair scan over a 1D array (up to 1024 elements).
/// Solves h_i = a_i * h_{i-1} + b_i (h_{-1} = 0).
public func pairScan(a: MLXArray, b: MLXArray) -> (MLXArray, MLXArray) {
    precondition(a.shape == b.shape, "a and b shapes must match")
    let n = a.size
    let outputs = MambaMetalKernels.pairScan(
        [a, b, MLXArray(UInt32(n))],
        grid: (1024, 1, 1),
        threadGroup: (1024, 1, 1),
        outputShapes: [a.shape, a.shape],
        outputDTypes: [a.dtype, a.dtype]
    )
    return (outputs[0], outputs[1])
}

/// Mamba selective scan (chunked) — runs the entire (batch, dim, seqlen) tensor
/// through the parallel scan kernel. Returns (y, final SSM state).
///
/// Shapes:
///   u, delta: (batch, dim, seqlen)
///   A:        (dim, dstate)
///   B, C:     (batch, dstate, seqlen)
///   D:        (dim,) — optional skip connection
///   z:        (batch, dim, seqlen) — optional SiLU gate
///
/// The kernel matches the Python mamba-metal implementation exactly.
public func selectiveScan(
    u: MLXArray, delta: MLXArray, A: MLXArray,
    B: MLXArray, C: MLXArray,
    D: MLXArray? = nil, z: MLXArray? = nil,
    deltaSoftplus: Bool = false
) -> (y: MLXArray, ssmState: MLXArray) {
    let batch = u.shape[0]
    let dim = u.shape[1]
    let seqlen = u.shape[2]
    let dstate = A.shape[1]

    let useD = D != nil
    let useZ = z != nil
    let dArg = D ?? MLX.zeros([dim], type: Float32.self)
    let zArg = z ?? MLX.zeros([1], type: Float32.self)

    let outputs = MambaMetalKernels.selectiveScanChunked(
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
