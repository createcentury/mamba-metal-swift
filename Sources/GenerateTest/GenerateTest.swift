// End-to-end: load mamba-130m-hf + tokenizer, greedy-generate text from a prompt.

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

        print("Loading from: \(dir)")

        let (model, _) = try loadMambaHF(
            safetensorsURL: URL(fileURLWithPath: "\(dir)/model.safetensors"),
            configURL: URL(fileURLWithPath: "\(dir)/config.json")
        )

        let tokenizer = try await AutoTokenizer.from(
            modelFolder: URL(fileURLWithPath: dir),
            strict: false   // GPTNeoXTokenizer falls through to BPETokenizer
        )

        let prompts = [
            "Mamba is a",
            "The capital of Japan is",
        ]

        for prompt in prompts {
            print("\n=== \(prompt.debugDescription) ===")
            print(prompt, terminator: "")
            fflush(stdout)
            let t0 = Date()
            _ = greedyGenerate(model: model, tokenizer: tokenizer, prompt: prompt, maxNewTokens: 30) { chunk in
                print(chunk, terminator: "")
                fflush(stdout)
            }
            let dt = Date().timeIntervalSince(t0)
            print("\n  [\(String(format: "%.2f", dt))s total, \(String(format: "%.1f", 30 / dt)) tok/s]")
        }
    }
}
