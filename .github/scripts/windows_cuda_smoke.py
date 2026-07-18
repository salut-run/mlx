"""Exercise the CUDA kernels Salut needs on a Windows Blackwell GPU."""

from __future__ import annotations

import os
import subprocess

import mlx.core as mx


def gpu_details() -> tuple[str, str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    name, compute_capability, driver, memory_mb = (
        value.strip() for value in output.splitlines()[0].split(",")
    )
    print(
        f"NVIDIA GPU: {name}; compute capability {compute_capability}; "
        f"driver {driver}; {memory_mb} MiB"
    )
    return name, compute_capability


def main() -> None:
    name, compute_capability = gpu_details()
    expected_name = os.environ.get("EXPECTED_GPU_NAME", "RTX 5070 Ti")
    if expected_name.casefold() not in name.casefold():
        raise RuntimeError(f"Expected {expected_name!r}, found {name!r}")
    if compute_capability != "12.0":
        raise RuntimeError(
            f"Expected Blackwell compute capability 12.0, found {compute_capability}"
        )

    device = mx.default_device()
    print(f"MLX device: {device}")
    if device != mx.gpu:
        raise RuntimeError(f"MLX did not select the CUDA GPU: {device}")

    mx.random.seed(0)

    left = mx.random.normal(shape=(256, 256)).astype(mx.float16)
    right = mx.random.normal(shape=(256, 256)).astype(mx.float16)
    product = left @ right

    activations = mx.random.normal(shape=(8, 128)).astype(mx.float16)
    weights = mx.random.normal(shape=(64, 128)).astype(mx.float16)
    quantized, scales, biases = mx.quantize(weights, group_size=64, bits=4)
    quantized_output = mx.quantized_matmul(
        activations,
        quantized,
        scales,
        biases,
        transpose=True,
        group_size=64,
        bits=4,
    )

    queries = mx.random.normal(shape=(1, 4, 16, 64)).astype(mx.float16)
    keys = mx.random.normal(shape=(1, 4, 16, 64)).astype(mx.float16)
    values = mx.random.normal(shape=(1, 4, 16, 64)).astype(mx.float16)
    attention = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=64**-0.5,
    )

    mx.eval(product, quantized_output, attention)
    for label, value in (
        ("matmul", product),
        ("4-bit quantized matmul", quantized_output),
        ("scaled dot-product attention", attention),
    ):
        if not mx.all(mx.isfinite(value)).item():
            raise RuntimeError(f"{label} produced non-finite values")
        print(f"{label}: shape={value.shape}, dtype={value.dtype}")

    print(f"Peak MLX memory: {mx.get_peak_memory() / 1024**2:.1f} MiB")
    print("Windows Blackwell CUDA smoke passed")


if __name__ == "__main__":
    main()
