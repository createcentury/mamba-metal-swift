// Greedy text generation for MambaModel.
// Simple O(L^2) variant: re-runs full forward each step (no state caching yet).

import Foundation
import MLX
import Tokenizers

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
