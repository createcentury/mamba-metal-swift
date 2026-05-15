// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mamba-metal-swift",
    platforms: [.macOS(.v14), .iOS(.v16)],
    products: [
        .library(name: "MambaMetal", targets: ["MambaMetal"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
    ],
    targets: [
        .target(
            name: "MambaMetal",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
            ],
            path: "Sources/MambaMetal"
        ),
        .executableTarget(
            name: "PairScanTest",
            dependencies: [
                .target(name: "MambaMetal"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/PairScanTest"
        ),
        .executableTarget(
            name: "SelectiveScanTest",
            dependencies: [
                .target(name: "MambaMetal"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/SelectiveScanTest"
        )
    ]
)
