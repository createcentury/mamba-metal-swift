// MambaMetalDemo — verifies our custom Metal scan kernel runs on iPhone GPU.

import SwiftUI
import MLX

@main
struct MambaMetalDemoApp: App {
    init() {
        // Keep MLX's cache small on iOS so it doesn't push us into jetsam.
        // 128 MB is a balance between perf (some cache helps) and footprint.
        MLX.Memory.cacheLimit = 128 * 1024 * 1024
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
