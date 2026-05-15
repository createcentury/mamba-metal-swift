// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mamba-metal-swift",
    platforms: [.macOS(.v14), .iOS(.v16)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
    ],
    targets: [
        .executableTarget(
            name: "PairScanTest",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/PairScanTest"
        )
    ]
)
