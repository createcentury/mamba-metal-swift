// End-to-end: load mamba-130m-hf + tokenizer, generate text both ways.
// Slow path = O(L^2) full forward each step; fast path = prefill + step state caching.
// Greedy outputs must match.

import Foundation
import MLX
import Tokenizers
import MambaMetal

@main
struct GenerateTest {
    static func main() async throws {
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        let snapBase = "\(home)/.cache/huggingface/hub/models--state-spaces--mamba-130m-hf/snapshots"
        let dirs = try FileManager.default.contentsOfDirectory(atPath: snapBase)
        guard let snap = dirs.first else { fatalError("no snapshot") }
        let dir = "\(snapBase)/\(snap)"

        print("Loading mamba-130m-hf …")
        let (model, _) = try loadMambaHF(
            safetensorsURL: URL(fileURLWithPath: "\(dir)/model.safetensors"),
            configURL: URL(fileURLWithPath: "\(dir)/config.json")
        )
        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: dir),
            strict: false
        )

        let prompts = ["The capital of Japan is"]
        let n = 30
        for prompt in prompts {
            print("\n=== \(prompt) ===")
            let t0 = Date()
            let slow = greedyGenerate(model: model, tokenizer: tokenizer,
                                       prompt: prompt, maxNewTokens: n)
            let tSlow = Date().timeIntervalSince(t0)

            let t1 = Date()
            let fast = greedyGenerateFast(model: model, tokenizer: tokenizer,
                                           prompt: prompt, maxNewTokens: n)
            let tFast = Date().timeIntervalSince(t1)

            print("[slow O(L²)] \(String(format: "%.2f", tSlow))s  \(String(format: "%.1f", Double(n)/tSlow)) tok/s")
            print("[fast O(L) ] \(String(format: "%.2f", tFast))s  \(String(format: "%.1f", Double(n)/tFast)) tok/s")
            print("[speedup   ] \(String(format: "%.2fx", tSlow / tFast))")
            print("[slow] \(slow)")
            print("[fast] \(fast)")
            if slow == fast {
                print("✅ outputs identical")
            } else {
                print("❌ outputs diverge")
            }
        }
    }
}
