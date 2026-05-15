import SwiftUI
import MLX
import MambaMetal
import Hub
import Tokenizers

enum SelectedModel: String, CaseIterable, Identifiable {
    case mamba = "Mamba 130m"
    case transformer = "SmolLM 135M (4-bit)"
    var id: String { rawValue }
}

struct ContentView: View {
    @State private var prompt: String = "The capital of Japan is"
    @State private var output: String = ""
    @State private var status: String = "Not loaded."
    @State private var isLoading: Bool = false
    @State private var isGenerating: Bool = false
    @State private var model: MambaModel?
    @State private var tokenizer: Tokenizer?
    @State private var tokensPerSec: Double = 0
    @State private var maxNewTokens: Double = 50
    @State private var useFastDecode: Bool = true
    @State private var sweepStatus: String = ""
    @State private var lastSweepPath: String = ""
    @State private var selectedModel: SelectedModel = .mamba
    @State private var transformer = TransformerRunner()

    private var modelReady: Bool {
        switch selectedModel {
        case .mamba: return model != nil && tokenizer != nil
        case .transformer: return transformer.loaded
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Model") {
                    Picker("model", selection: $selectedModel) {
                        ForEach(SelectedModel.allCases) { m in
                            Text(m.rawValue).tag(m)
                        }
                    }
                    .disabled(isLoading || isGenerating)
                    Text(status).font(.footnote).foregroundStyle(.secondary)
                    Button(action: { Task { await loadModel() } }) {
                        Text(selectedModel == .mamba
                             ? (model == nil ? "Download & Load mamba-130m" : "Reload Mamba")
                             : (transformer.loaded ? "Reload Transformer" : "Download & Load SmolLM-135M"))
                    }
                    .disabled(isLoading || isGenerating)
                }
                Section("Prompt") {
                    TextField("prompt", text: $prompt, axis: .vertical)
                        .lineLimit(2...5)
                    HStack {
                        Text("max new tokens: \(Int(maxNewTokens))")
                            .font(.footnote)
                        Spacer()
                    }
                    Slider(value: $maxNewTokens, in: 10...400, step: 10)
                    Toggle("fast decode (prefill + step, O(L))", isOn: $useFastDecode)
                        .font(.footnote)
                    Button(isGenerating ? "Generating…" : "Generate") {
                        Task { await generate() }
                    }
                    .disabled(!modelReady || isGenerating)
                }
                Section("Output") {
                    ScrollView {
                        Text(output)
                            .font(.system(.body, design: .serif))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 300)
                    if tokensPerSec > 0 {
                        Text(String(format: "%.1f tokens/sec", tokensPerSec))
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                Section("Auto sweep") {
                    Button(selectedModel == .mamba
                           ? "Run (50, 200, 400) × (fast, slow)"
                           : "Run (50, 200, 400) tokens") {
                        Task { await runSweep() }
                    }
                    .disabled(!modelReady || isGenerating)
                    if !sweepStatus.isEmpty {
                        Text(sweepStatus).font(.footnote).foregroundStyle(.secondary)
                    }
                    if !lastSweepPath.isEmpty {
                        Text("→ saved: \(lastSweepPath)")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundStyle(.secondary)
                    }
                }
                Section("Custom kernel sanity") {
                    NavigationLink("Run pair_scan / selective_scan smoke tests") {
                        KernelSmokeView()
                    }
                }
            }
            .navigationTitle("mamba-metal")
        }
    }

    private func loadModel() async {
        isLoading = true
        defer { isLoading = false }
        do {
            if selectedModel == .mamba {
                status = "Downloading mamba-130m-hf …"
                let hub = HubApi()
                let repo = Hub.Repo(id: "state-spaces/mamba-130m-hf")
                let folder = try await hub.snapshot(from: repo, matching: ["*.json", "*.safetensors"]) { p in
                    Task { @MainActor in status = String(format: "Downloading … %.0f%%", p.fractionCompleted * 100) }
                }
                status = "Loading weights …"
                let (m, _) = try loadMambaHF(
                    safetensorsURL: folder.appendingPathComponent("model.safetensors"),
                    configURL: folder.appendingPathComponent("config.json")
                )
                let t = try await AutoTokenizer.from(modelFolder: folder, strict: false)
                self.model = m
                self.tokenizer = t
                status = "✅ mamba-130m loaded."
            } else {
                status = "Downloading SmolLM-135M-4bit …"
                try await transformer.load { p in
                    status = String(format: "Downloading … %.0f%%", p * 100)
                }
                status = "✅ SmolLM-135M loaded."
            }
        } catch {
            status = "❌ load failed: \(error)"
        }
    }

    private func generate() async {
        guard let model, let tokenizer else { return }
        isGenerating = true
        defer { isGenerating = false }
        output = prompt
        tokensPerSec = 0
        let start = Date()
        var count = 0

        await runOnce(maxTokens: Int(maxNewTokens), fast: useFastDecode)
    }

    private struct SweepResult: Codable {
        let device: String
        let model: String
        let framework: String
        let prompt: String
        let promptTokens: Int
        let runs: [Run]
        let timestamp: String

        struct Run: Codable {
            let maxTokens: Int
            let fast: Bool
            let totalSeconds: Double
            let tokensPerSec: Double
            let physFootprintMBPeak: Double
        }
    }

    private func runSweep() async {
        isGenerating = true
        defer { isGenerating = false }

        // For Transformer side: fast/slow distinction doesn't apply, only run fast variants.
        // Apples-to-apples: both models use the same fast decode path.
        // n_new spans 50 → 1000 to surface the KV-cache cost curve on the Transformer.
        let configs: [(Int, Bool)] = [
            (50, true), (200, true), (400, true), (1000, true)
        ]
        if selectedModel == .mamba {
            guard model != nil, tokenizer != nil else { return }
        } else {
            guard transformer.loaded else { return }
        }
        var runs: [SweepResult.Run] = []
        for (i, (n, fast)) in configs.enumerated() {
            // Free any cached MLX buffers from the previous run.
            MLX.Memory.clearCache()
            sweepStatus = "Running \(i+1)/\(configs.count): max=\(n) \(fast ? "fast" : "slow")…"
            await Task.yield()
            let t0 = Date()
            let (peakMB, _) = await runOnceMeasured(maxTokens: n, fast: fast)
            let dt = Date().timeIntervalSince(t0)
            runs.append(.init(
                maxTokens: n, fast: fast,
                totalSeconds: dt,
                tokensPerSec: Double(n) / dt,
                physFootprintMBPeak: peakMB
            ))
            // Persist partial results after each run so we don't lose them on crash.
            writePartial(runs: runs)
        }

        let promptTokens = (selectedModel == .mamba) ? (tokenizer?.encode(text: prompt).count ?? 0) : -1
        let result = SweepResult(
            device: "iPhone Air (iPhone18,4)",
            model: selectedModel == .mamba ? "state-spaces/mamba-130m-hf" : "mlx-community/SmolLM-135M-Instruct-4bit",
            framework: selectedModel == .mamba ? "mamba-metal-swift" : "mlx-swift-lm",
            prompt: prompt,
            promptTokens: promptTokens,
            runs: runs,
            timestamp: ISO8601DateFormatter().string(from: Date())
        )

        // Write to Documents/sweep-<timestamp>.json
        let docs = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true
        )
        if let docs {
            let safeTs = result.timestamp.replacingOccurrences(of: ":", with: "-")
            let url = docs.appendingPathComponent("sweep-\(safeTs).json")
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? enc.encode(result) {
                try? data.write(to: url)
                lastSweepPath = "Documents/sweep-\(safeTs).json"
            }
        }
        sweepStatus = "Sweep done (\(runs.count) runs)."
    }

    private func writePartial(runs: [SweepResult.Run]) {
        let promptTokens = (selectedModel == .mamba) ? (tokenizer?.encode(text: prompt).count ?? 0) : -1
        let modelTag = selectedModel == .mamba ? "mamba-130m" : "smollm-135m"
        let result = SweepResult(
            device: "iPhone Air (iPhone18,4)",
            model: selectedModel == .mamba ? "state-spaces/mamba-130m-hf" : "mlx-community/SmolLM-135M-Instruct-4bit",
            framework: selectedModel == .mamba ? "mamba-metal-swift" : "mlx-swift-lm",
            prompt: prompt, promptTokens: promptTokens,
            runs: runs,
            timestamp: ISO8601DateFormatter().string(from: Date())
        )
        if let docs = try? FileManager.default.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true) {
            let url = docs.appendingPathComponent("sweep-\(modelTag).json")
            let enc = JSONEncoder()
            enc.outputFormatting = [.prettyPrinted, .sortedKeys]
            if let data = try? enc.encode(result) {
                try? data.write(to: url)
                lastSweepPath = "Documents/sweep-\(modelTag).json"
            }
        }
    }

    private func runOnce(maxTokens n: Int, fast: Bool) async {
        let (_, _) = await runOnceMeasured(maxTokens: n, fast: fast)
    }

    private func runOnceMeasured(maxTokens n: Int, fast: Bool) async -> (peakMB: Double, text: String) {
        output = prompt
        tokensPerSec = 0
        let start = Date()
        var count = 0
        var peakMB = 0.0

        // Transformer branch
        if selectedModel == .transformer {
            do {
                let (_, _) = try await transformer.generate(prompt: prompt, maxNewTokens: n) { chunk in
                    count += 1
                    output += chunk
                    let elapsed = Date().timeIntervalSince(start)
                    if elapsed > 0 { tokensPerSec = Double(count) / elapsed }
                    let mem = currentPhysFootprintMB()
                    if mem > peakMB { peakMB = mem }
                }
            } catch {
                output += "\n[error: \(error)]"
            }
            return (peakMB, output)
        }

        // Mamba branch
        guard let model, let tokenizer else { return (0, "") }
        await Task.detached {
            let cb: (String) -> Void = { chunk in
                count += 1
                // Sample memory
                let mem = currentPhysFootprintMB()
                if mem > peakMB { peakMB = mem }
                Task { @MainActor in
                    output += chunk
                    let elapsed = Date().timeIntervalSince(start)
                    if elapsed > 0 { tokensPerSec = Double(count) / elapsed }
                }
            }
            if fast {
                _ = greedyGenerateFast(
                    model: model, tokenizer: tokenizer,
                    prompt: prompt, maxNewTokens: n, onToken: cb)
            } else {
                _ = greedyGenerate(
                    model: model, tokenizer: tokenizer,
                    prompt: prompt, maxNewTokens: n, onToken: cb)
            }
        }.value
        return (peakMB, output)
    }
}

import Darwin
fileprivate func currentPhysFootprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
        MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size
    )
    let result = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    if result == KERN_SUCCESS {
        return Double(info.phys_footprint) / (1024.0 * 1024.0)
    }
    return 0
}

// Existing kernel sanity from earlier version, moved into a separate view.
struct KernelSmokeView: View {
    @State private var status: String = "Tap a button to run."
    @State private var detail: String = ""
    @State private var elapsed: String = ""

    var body: some View {
        Form {
            Section("Status") {
                Text(status).font(.headline)
                Text(detail).font(.system(.body, design: .monospaced)).foregroundStyle(.secondary)
                Text(elapsed).font(.footnote)
            }
            Section {
                Button("Run pair_scan") { runPairScan() }
                Button("Run selective_scan (B=1, D=2, N=8, T=256)") { runSelectiveScan() }
            }
        }
        .navigationTitle("Kernel smoke")
    }

    private func runPairScan() {
        let n = 1024
        let aHost = [Float](repeating: 0.5, count: n)
        let bHost = [Float](repeating: 1.0, count: n)
        let t0 = Date()
        let (_, hOut) = pairScan(a: MLXArray(aHost, [n]), b: MLXArray(bHost, [n]))
        eval(hOut)
        let dt = Date().timeIntervalSince(t0)
        let arr = hOut.asArray(Float.self)
        var maxErr: Float = 0
        var h: Float = 0
        for i in 0..<n {
            h = 0.5 * h + 1.0
            maxErr = max(maxErr, abs(arr[i] - h))
        }
        status = maxErr < 1e-4 ? "✅ pair_scan OK" : "❌ mismatch"
        detail = "n=\(n) max abs err = \(String(format: "%.2e", maxErr))"
        elapsed = String(format: "%.1f ms", dt * 1000)
    }

    private func runSelectiveScan() {
        let batch = 1, dim = 2, dstate = 8, seqlen = 256
        var seed: UInt32 = 12345
        func lcg() -> Float {
            seed = seed &* 1664525 &+ 1013904223
            return Float(seed) / Float(UInt32.max) * 2.0 - 1.0
        }
        let u = (0..<(batch*dim*seqlen)).map { _ in lcg() }
        let delta = (0..<(batch*dim*seqlen)).map { _ in 0.01 + 0.09 * (lcg() + 1.0) / 2.0 }
        let A = (0..<(dim*dstate)).map { _ in -(0.1 + 1.9 * (lcg() + 1.0) / 2.0) }
        let B = (0..<(batch*dstate*seqlen)).map { _ in lcg() }
        let C = (0..<(batch*dstate*seqlen)).map { _ in lcg() }

        let t0 = Date()
        let (yMx, _) = selectiveScan(
            u: MLXArray(u, [batch, dim, seqlen]),
            delta: MLXArray(delta, [batch, dim, seqlen]),
            A: MLXArray(A, [dim, dstate]),
            B: MLXArray(B, [batch, dstate, seqlen]),
            C: MLXArray(C, [batch, dstate, seqlen])
        )
        eval(yMx)
        let dt = Date().timeIntervalSince(t0)

        var yRef = [Float](repeating: 0, count: batch*dim*seqlen)
        for bi in 0..<batch {
            for di in 0..<dim {
                var h = [Float](repeating: 0, count: dstate)
                for t in 0..<seqlen {
                    let dtv = delta[bi*dim*seqlen + di*seqlen + t]
                    let ut = u[bi*dim*seqlen + di*seqlen + t]
                    var ys: Float = 0
                    for s in 0..<dstate {
                        let At = exp(dtv * A[di*dstate + s])
                        let bt = dtv * ut * B[bi*dstate*seqlen + s*seqlen + t]
                        h[s] = At * h[s] + bt
                        ys += h[s] * C[bi*dstate*seqlen + s*seqlen + t]
                    }
                    yRef[bi*dim*seqlen + di*seqlen + t] = ys
                }
            }
        }
        let yMetal = yMx.asArray(Float.self)
        var maxErr: Float = 0
        for i in 0..<yRef.count { maxErr = max(maxErr, abs(yMetal[i] - yRef[i])) }
        status = maxErr < 1e-4 ? "✅ selective_scan OK" : "❌ mismatch"
        detail = "B=\(batch) D=\(dim) N=\(dstate) T=\(seqlen) max abs err = \(String(format: "%.2e", maxErr))"
        elapsed = String(format: "%.1f ms", dt * 1000)
    }
}
