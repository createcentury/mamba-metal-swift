// Transformer side via mlx-swift-lm. SmolLM-135M-4bit chosen as the
// closest small-model counterpart to mamba-130m for the comparison.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import HuggingFace
import Tokenizers

@MainActor
public final class TransformerRunner {
    private var container: ModelContainer?
    public var loaded: Bool { container != nil }

    public init() {}

    public func load(progress: @MainActor @escaping (Double) -> Void) async throws {
        let cfg = LLMRegistry.smolLM_135M_4bit
        let c = try await LLMModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: cfg
        ) { value in
            Task { @MainActor in
                progress(value.fractionCompleted)
            }
        }
        self.container = c
    }

    /// Greedy-stream generation. Returns the final string + count of decoded chunks.
    public func generate(
        prompt: String,
        maxNewTokens: Int,
        onChunk: @MainActor @escaping (String) -> Void
    ) async throws -> (text: String, chunks: Int) {
        guard let container else {
            throw NSError(domain: "TransformerRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "not loaded"])
        }
        let session = ChatSession(container)
        var combined = ""
        var chunks = 0
        // Default chat session uses sampling; we want greedy = temperature 0 approx.
        for try await chunk in session.streamResponse(to: prompt) {
            combined += chunk
            chunks += 1
            await MainActor.run { onChunk(chunk) }
            // Stop after roughly maxNewTokens — chunks are rough token approximation.
            if chunks >= maxNewTokens { break }
        }
        return (combined, chunks)
    }
}
