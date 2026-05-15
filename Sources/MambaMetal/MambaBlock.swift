// Swift port of mamba_metal.mamba_block.MambaBlock (Python).
// Uses MLXNN's Linear / Conv1d / RMSNorm modules, with our custom
// selective scan Metal kernel for the SSM.

import Foundation
import MLX
import MLXNN

public class MambaBlock: Module {
    public let dModel: Int
    public let dState: Int
    public let dConv: Int
    public let dInner: Int
    public let dtRank: Int

    @ModuleInfo(key: "in_proj")   var inProj: Linear
    @ModuleInfo                   var conv1d: Conv1d
    @ModuleInfo(key: "x_proj")    var xProj: Linear
    @ModuleInfo(key: "dt_proj")   var dtProj: Linear
    @ModuleInfo(key: "A_log")     var ALog: MLXArray
    @ModuleInfo(key: "D")         var D: MLXArray
    @ModuleInfo(key: "out_proj")  var outProj: Linear

    public init(
        dModel: Int,
        dState: Int = 16,
        dConv: Int = 4,
        expand: Int = 2,
        dtRank: Int? = nil
    ) {
        self.dModel = dModel
        self.dState = dState
        self.dConv = dConv
        self.dInner = expand * dModel
        self.dtRank = dtRank ?? Int((Double(dModel) / 16.0).rounded(.up))

        self._inProj.wrappedValue = Linear(dModel, self.dInner * 2, bias: false)
        self._conv1d.wrappedValue = Conv1d(
            inputChannels: self.dInner,
            outputChannels: self.dInner,
            kernelSize: dConv,
            stride: 1,
            padding: dConv - 1,
            groups: self.dInner,
            bias: true
        )
        self._xProj.wrappedValue  = Linear(self.dInner, self.dtRank + 2 * dState, bias: false)
        self._dtProj.wrappedValue = Linear(self.dtRank, self.dInner, bias: true)

        // A_log such that A = -exp(A_log) is negative diagonal.
        // Init: A[d, n] = n+1, then log so the parameter is unconstrained.
        let aInit = MLX.broadcast(
            MLXArray(stride(from: 1, through: dState, by: 1).map { Float($0) }, [dState]),
            to: [self.dInner, dState]
        )
        self._ALog.wrappedValue = MLX.log(aInit)
        self._D.wrappedValue = MLX.ones([self.dInner])

        self._outProj.wrappedValue = Linear(self.dInner, dModel, bias: false)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, L, dModel)
        let L = x.shape[1]

        let xz = inProj(x)                            // (B, L, 2*dInner)
        let parts = xz.split(parts: 2, axis: -1)
        let xMainPre = parts[0]
        let z = parts[1]

        // Causal depth-wise conv: keep first L outputs only.
        var xMain = conv1d(xMainPre)
        xMain = xMain[0..., 0..<L, 0...]
        xMain = silu(xMain)

        let xDbl = xProj(xMain)                       // (B, L, dtRank + 2*dState)
        let dt   = xDbl[0..., 0..., 0 ..< dtRank]
        let bSsm = xDbl[0..., 0..., dtRank ..< (dtRank + dState)]
        let cSsm = xDbl[0..., 0..., (dtRank + dState) ..< (dtRank + 2 * dState)]
        let dtFull = dtProj(dt)                       // (B, L, dInner)

        let A = -MLX.exp(ALog)                        // (dInner, dState)

        let (yChan, _) = selectiveScan(
            u: xMain.transposed(0, 2, 1),
            delta: dtFull.transposed(0, 2, 1),
            A: A,
            B: bSsm.transposed(0, 2, 1),
            C: cSsm.transposed(0, 2, 1),
            D: D,
            z: z.transposed(0, 2, 1),
            deltaSoftplus: true
        )
        let y = yChan.transposed(0, 2, 1)             // (B, L, dInner)
        return outProj(y)
    }

    /// Process a prompt and return per-state caches for incremental decode.
    /// Returns (y, convState, ssmState).
    public func prefill(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let B = x.shape[0]
        let L = x.shape[1]

        let xz = inProj(x)
        let parts = xz.split(parts: 2, axis: -1)
        let xPreConv = parts[0]
        let z = parts[1]

        // Conv state: last d_conv pre-conv values (left-padded with zeros).
        let convState: MLXArray
        if L >= dConv {
            convState = xPreConv[0..., (L - dConv)..<L, 0...].transposed(0, 2, 1)
        } else {
            let pad = MLX.zeros([B, dConv - L, dInner], type: Float32.self)
            convState = MLX.concatenated([pad, xPreConv], axis: 1).transposed(0, 2, 1)
        }

        var xMain = conv1d(xPreConv)[0..., 0..<L, 0...]
        xMain = silu(xMain)

        let xDbl = xProj(xMain)
        let dt   = xDbl[0..., 0..., 0 ..< dtRank]
        let bSsm = xDbl[0..., 0..., dtRank ..< (dtRank + dState)]
        let cSsm = xDbl[0..., 0..., (dtRank + dState) ..< (dtRank + 2 * dState)]
        let dtFull = dtProj(dt)
        let A = -MLX.exp(ALog)

        let (yChan, ssmState) = selectiveScan(
            u: xMain.transposed(0, 2, 1),
            delta: dtFull.transposed(0, 2, 1),
            A: A,
            B: bSsm.transposed(0, 2, 1),
            C: cSsm.transposed(0, 2, 1),
            D: D,
            z: z.transposed(0, 2, 1),
            deltaSoftplus: true
        )
        let y = outProj(yChan.transposed(0, 2, 1))
        return (y, convState, ssmState)
    }

    /// O(1) single-token step. Returns (y, newConvState, newSSMState).
    public func step(
        _ xToken: MLXArray,         // (B, 1, dModel)
        convState: MLXArray,        // (B, dInner, dConv)
        ssmState: MLXArray          // (B, dInner, dState)
    ) -> (MLXArray, MLXArray, MLXArray) {
        let xz = inProj(xToken)
        let parts = xz.split(parts: 2, axis: -1)
        let xMain = parts[0][0..., 0, 0...]           // (B, dInner)
        let z = parts[1][0..., 0, 0...]               // (B, dInner)

        // Sliding-window conv update
        let newConv = MLX.concatenated(
            [convState[0..., 0..., 1..<dConv], xMain.expandedDimensions(axis: 2)],
            axis: 2
        )                                              // (B, dInner, dConv)
        let w = conv1d.weight.squeezed(axis: -1)       // (dInner, dConv)
        let convOut = (newConv * w.expandedDimensions(axis: 0)).sum(axis: -1)
            + (conv1d.bias ?? MLX.zeros([dInner], type: Float32.self))
        let xConv = silu(convOut)                     // (B, dInner)

        let xDbl = xProj(xConv)
        let dtPre = xDbl[0..., 0 ..< dtRank]
        let bSsm  = xDbl[0..., dtRank ..< (dtRank + dState)]
        let cSsm  = xDbl[0..., (dtRank + dState) ..< (dtRank + 2 * dState)]
        var dt = dtProj(dtPre)                        // (B, dInner)
        dt = softplus(dt)

        let A = -MLX.exp(ALog)                        // (dInner, dState)
        let a = MLX.exp(
            dt.expandedDimensions(axis: 2) * A.expandedDimensions(axis: 0)
        )                                              // (B, dInner, dState)
        let b = (dt * xConv).expandedDimensions(axis: 2)
            * bSsm.expandedDimensions(axis: 1)         // (B, dInner, dState)
        let newSSM = a * ssmState + b                 // (B, dInner, dState)

        var y = (newSSM * cSsm.expandedDimensions(axis: 1)).sum(axis: -1)  // (B, dInner)
        y = y + D * xConv                              // D skip
        y = y * silu(z)                                // z gate

        y = outProj(y).expandedDimensions(axis: 1)    // (B, 1, dModel)
        return (y, newConv, newSSM)
    }
}
