// Greedy text generation for MambaModel.
// Simple O(L^2) variant: re-runs full forward each step (no state caching yet).

import Foundation
import MLX
import Tokenizers

/// O(L_prompt + n_decode) generation using prefill + step state caching.
public func greedyGenerateFast(
    model: MambaModel,
    tokenizer: Tokenizer,
    prompt: String,
    maxNewTokens: Int = 50,
    onToken: ((String) -> Void)? = nil
) -> String {
    let promptIds = tokenizer.encode(text: prompt).map { Int32($0) }
    let inputIds = MLXArray(promptIds, [1, promptIds.count])

    var (logits, conv, ssm) = model.prefill(inputIds)
    eval(logits)

    let eos = tokenizer.eosTokenId
    var generated: [Int] = []
    var prevText = prompt
    var allIds = promptIds.map { Int($0) }

    for _ in 0..<maxNewTokens {
        let last = logits[0..., -1, 0...]
        let next = MLX.argMax(last, axis: -1)
        eval(next)
        let nextId = Int(next.asArray(Int32.self)[0])
        if let e = eos, nextId == e { break }
        generated.append(nextId)
        allIds.append(nextId)
        if let cb = onToken {
            let nowText = tokenizer.decode(tokens: allIds)
            let delta = String(nowText.dropFirst(prevText.count))
            if !delta.isEmpty { cb(delta); prevText = nowText }
        }
        let step = model.step(
            MLXArray([Int32(nextId)], [1, 1]),
            conv: conv, ssm: ssm
        )
        logits = step.logits
        conv = step.conv
        ssm = step.ssm
        eval(logits)
    }
    return prompt + tokenizer.decode(tokens: generated)
}

public func greedyGenerate(
    model: MambaModel,
    tokenizer: Tokenizer,
    prompt: String,
    maxNewTokens: Int = 50,
    onToken: ((String) -> Void)? = nil
) -> String {
    var ids = tokenizer.encode(text: prompt)
    let eos = tokenizer.eosTokenId
    var generated: [Int] = []
    var prevText = prompt

    for _ in 0..<maxNewTokens {
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let logits = model(inputIds)
        let lastLogits = logits[0..., -1, 0...]
        let next = MLX.argMax(lastLogits, axis: -1)
        eval(next)
        let nextId = Int(next.asArray(Int32.self)[0])
        if eos != nil && nextId == eos { break }
        generated.append(nextId)
        ids.append(nextId)

        if let cb = onToken {
            let nowText = tokenizer.decode(tokens: ids)
            let delta = String(nowText.dropFirst(prevText.count))
            if !delta.isEmpty {
                cb(delta)
                prevText = nowText
            }
        }
    }
    return prompt + tokenizer.decode(tokens: generated)
}
