// Load a state-spaces/mamba-*-hf checkpoint into MambaModel.
// Same two transforms as the Python project:
//   - Strip "backbone." prefix from every key
//   - Transpose conv1d.weight from PyTorch (out, in/g, k) to MLX (out, k, in/g)
//
// Pass paths to model.safetensors + config.json (e.g. the HF cache directory).

import Foundation
import MLX
import MLXNN

public struct MambaConfigJSON: Decodable {
    public let hidden_size: Int?
    public let d_model: Int?
    public let intermediate_size: Int?
    public let d_inner: Int?
    public let num_hidden_layers: Int?
    public let n_layer: Int?
    public let vocab_size: Int
    public let state_size: Int?
    public let d_state: Int?
    public let conv_kernel: Int?
    public let d_conv: Int?
    public let time_step_rank: Int?
    public let dt_rank: Int?
    public let expand: Int?
    public let layer_norm_epsilon: Float?

    public func toConfig() -> MambaConfig {
        let dModel = hidden_size ?? d_model!
        let dInner = intermediate_size ?? d_inner ?? (2 * dModel)
        let expand = self.expand ?? (dInner / dModel)
        return MambaConfig(
            dModel: dModel,
            nLayer: num_hidden_layers ?? n_layer!,
            vocabSize: vocab_size,
            dState: state_size ?? d_state ?? 16,
            dConv: conv_kernel ?? d_conv ?? 4,
            expand: expand,
            dtRank: time_step_rank ?? dt_rank,
            rmsNormEps: layer_norm_epsilon ?? 1e-5
        )
    }
}

public func loadMambaHF(safetensorsURL: URL, configURL: URL) throws -> (MambaModel, MambaConfig) {
    let configData = try Data(contentsOf: configURL)
    let cfgJSON = try JSONDecoder().decode(MambaConfigJSON.self, from: configData)
    let cfg = cfgJSON.toConfig()

    let raw = try MLX.loadArrays(url: safetensorsURL)
    let mapped = transformKeys(raw)

    let model = MambaModel(cfg)
    // mlx-swift's Module.update(parameters:) expects a NestedDictionary; convert flat -> nested
    let nested = ModuleParameters.unflattened(mapped)
    let _ = try model.update(parameters: nested, verify: [.noUnusedKeys])
    eval(model.parameters())
    return (model, cfg)
}

/// Transform HF keys to ours:
///   "backbone.layers.0.mixer.in_proj.weight" -> "layers.0.mixer.in_proj.weight"
/// And transpose conv1d.weight (out, in/g, k) -> (out, k, in/g).
public func transformKeys(_ raw: [String: MLXArray]) -> [String: MLXArray] {
    var out = [String: MLXArray]()
    out.reserveCapacity(raw.count)
    for (k, v) in raw {
        var name = k
        if name.hasPrefix("backbone.") {
            name = String(name.dropFirst("backbone.".count))
        }
        if name.hasSuffix("conv1d.weight") {
            out[name] = v.transposed(0, 2, 1)
        } else {
            out[name] = v
        }
    }
    return out
}
