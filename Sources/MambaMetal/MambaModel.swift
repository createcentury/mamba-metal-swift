// Swift port of mamba_metal.mamba_model — full Mamba causal LM stack.

import Foundation
import MLX
import MLXNN

public struct MambaConfig {
    public let dModel: Int
    public let nLayer: Int
    public let vocabSize: Int
    public let dState: Int
    public let dConv: Int
    public let expand: Int
    public let dtRank: Int?
    public let rmsNormEps: Float

    public init(
        dModel: Int = 768,
        nLayer: Int = 24,
        vocabSize: Int = 50280,
        dState: Int = 16,
        dConv: Int = 4,
        expand: Int = 2,
        dtRank: Int? = nil,
        rmsNormEps: Float = 1e-5
    ) {
        self.dModel = dModel
        self.nLayer = nLayer
        self.vocabSize = vocabSize
        self.dState = dState
        self.dConv = dConv
        self.expand = expand
        self.dtRank = dtRank
        self.rmsNormEps = rmsNormEps
    }
}

public class MambaResidualBlock: Module {
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo var mixer: MambaBlock

    public init(_ cfg: MambaConfig) {
        self._norm.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: cfg.rmsNormEps)
        self._mixer.wrappedValue = MambaBlock(
            dModel: cfg.dModel,
            dState: cfg.dState,
            dConv: cfg.dConv,
            expand: cfg.expand,
            dtRank: cfg.dtRank
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return x + mixer(norm(x))
    }
}

public class MambaModel: Module {
    public let cfg: MambaConfig

    @ModuleInfo var embeddings: Embedding
    @ModuleInfo var layers: [MambaResidualBlock]
    @ModuleInfo(key: "norm_f") var normF: RMSNorm

    public init(_ cfg: MambaConfig) {
        self.cfg = cfg
        self._embeddings.wrappedValue = Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.dModel)
        self._layers.wrappedValue = (0..<cfg.nLayer).map { _ in MambaResidualBlock(cfg) }
        self._normF.wrappedValue = RMSNorm(dimensions: cfg.dModel, eps: cfg.rmsNormEps)
        super.init()
    }

    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        // inputIds: (B, L), int — returns logits (B, L, vocabSize)
        var x = embeddings(inputIds)
        for layer in layers {
            x = layer(x)
        }
        x = normF(x)
        // Tied LM head: logits = x @ embeddings.weight.T
        return x.matmul(embeddings.weight.T)
    }

    public func initState(batchSize: Int = 1) -> (conv: [MLXArray], ssm: [MLXArray]) {
        let dInner = cfg.expand * cfg.dModel
        let conv = (0..<cfg.nLayer).map { _ in
            MLX.zeros([batchSize, dInner, cfg.dConv], type: Float32.self)
        }
        let ssm = (0..<cfg.nLayer).map { _ in
            MLX.zeros([batchSize, dInner, cfg.dState], type: Float32.self)
        }
        return (conv, ssm)
    }

    /// Run the entire prompt once. Returns logits + per-layer state for decode continuation.
    public func prefill(_ inputIds: MLXArray)
        -> (logits: MLXArray, conv: [MLXArray], ssm: [MLXArray])
    {
        var x = embeddings(inputIds)
        var conv: [MLXArray] = []
        var ssm: [MLXArray] = []
        for layer in layers {
            let (y, cs, ss) = layer.mixer.prefill(layer.norm(x))
            x = x + y
            conv.append(cs)
            ssm.append(ss)
        }
        x = normF(x)
        return (x.matmul(embeddings.weight.T), conv, ssm)
    }

    /// O(1) per-token step. Carries state across calls.
    public func step(
        _ inputIds: MLXArray,
        conv: [MLXArray], ssm: [MLXArray]
    ) -> (logits: MLXArray, conv: [MLXArray], ssm: [MLXArray]) {
        var x = embeddings(inputIds)
        var newConv: [MLXArray] = []
        var newSSM: [MLXArray] = []
        for (i, layer) in layers.enumerated() {
            let (y, cs, ss) = layer.mixer.step(layer.norm(x), convState: conv[i], ssmState: ssm[i])
            x = x + y
            newConv.append(cs)
            newSSM.append(ss)
        }
        x = normF(x)
        return (x.matmul(embeddings.weight.T), newConv, newSSM)
    }
}
