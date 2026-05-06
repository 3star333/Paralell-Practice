"""
Lab 4 — ReLU: Triton vs Torch
Implements ReLU as a Triton kernel and benchmarks it against torch.relu
across a range of input sizes, producing a GB/s throughput plot.
"""

import torch
import triton
import triton.language as tl
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────
# Triton ReLU kernel
# Each program instance handles a contiguous block of BLOCK_SIZE
# elements. The mask ensures safe handling of sizes that are not
# divisible by BLOCK_SIZE (boundary condition).
# ─────────────────────────────────────────────────────────────
@triton.jit
def relu_kernel(
    x_ptr,          # pointer to input  tensor
    y_ptr,          # pointer to output tensor
    n_elements,     # total number of elements
    BLOCK_SIZE: tl.constexpr,  # number of elements per program instance
):
    # each program handles a unique block of elements
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # boundary mask — guards against out-of-bounds for non-power-of-2 sizes
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)            # ReLU: max(x, 0)
    tl.store(y_ptr + offsets, y, mask=mask)


# ─────────────────────────────────────────────────────────────
# Host wrapper — allocates output, launches kernel
# ─────────────────────────────────────────────────────────────
def triton_relu(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024   # threads per block — power of 2, fits in shared budget

    # 1-D grid: ceil(n_elements / BLOCK_SIZE) programs
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return y


# ─────────────────────────────────────────────────────────────
# Correctness check
# ─────────────────────────────────────────────────────────────
def check_correctness():
    torch.manual_seed(0)
    x = torch.randn(1024 * 1024, device='cuda', dtype=torch.float32)

    y_triton = triton_relu(x)
    y_torch  = torch.relu(x)

    max_diff = (y_triton - y_torch).abs().max().item()
    match    = torch.allclose(y_triton, y_torch, atol=1e-6)

    print("=" * 52)
    print("Correctness check")
    print("=" * 52)
    print(f"  Max absolute difference : {max_diff:.2e}")
    print(f"  Results match           : {'PASSED ✓' if match else 'FAILED ✗'}")
    print()


# ─────────────────────────────────────────────────────────────
# Benchmark
# triton.testing.Benchmark measures GB/s throughput.
# Each element is read once and written once → 2 * size bytes.
# ─────────────────────────────────────────────────────────────
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names  = ['size'],
        x_vals   = [2**i for i in range(12, 28)],   # 4K → 128M elements
        x_log    = True,
        line_arg = 'provider',
        line_vals= ['triton', 'torch'],
        line_names=['Triton ReLU', 'Torch ReLU'],
        styles   = [('royalblue', '-o'), ('tomato', '-s')],
        ylabel   = 'Bandwidth (GB/s)',
        plot_name= 'ReLU Performance: Triton vs Torch',
        args     = {},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]   # median + 20th/80th percentile bands

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: triton_relu(x), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.relu(x), quantiles=quantiles)

    # throughput = bytes read + bytes written  /  time
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    check_correctness()

    print("=" * 52)
    print("Benchmark (GB/s throughput)")
    print("=" * 52)

    # run benchmark, save plot and print table
    benchmark.run(
        print_data=True,
        show_plots=False,
        save_path='.'          # saves 'ReLU Performance: Triton vs Torch.png'
    )

    # ── re-draw in a clean style matching the Triton tutorial ──
    sizes = [2**i for i in range(12, 28)]
    labels = [f'2^{i}' if i % 2 == 0 else '' for i in range(12, 28)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title('ReLU Performance: Triton vs Torch', fontsize=14, fontweight='bold')
    ax.set_xlabel('Input size (number of float32 elements)', fontsize=11)
    ax.set_ylabel('Bandwidth (GB/s)', fontsize=11)
    ax.set_xscale('log', base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f'$2^{{{i}}}$' for i in range(12, 28)], fontsize=8)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)

    # collect results for both providers
    results = {}
    for provider, color, marker, label in [
        ('triton', 'royalblue', 'o', 'Triton ReLU'),
        ('torch',  'tomato',    's', 'Torch ReLU'),
    ]:
        gbps_vals, gbps_max, gbps_min = [], [], []
        for size in sizes:
            x = torch.randn(size, device='cuda', dtype=torch.float32)
            quantiles = [0.5, 0.2, 0.8]
            if provider == 'triton':
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: triton_relu(x), quantiles=quantiles)
            else:
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: torch.relu(x), quantiles=quantiles)
            to_gbps = lambda t: 2 * x.numel() * x.element_size() * 1e-9 / (t * 1e-3)
            gbps_vals.append(to_gbps(ms))
            gbps_min.append(to_gbps(max_ms))
            gbps_max.append(to_gbps(min_ms))

        ax.plot(sizes, gbps_vals, color=color, marker=marker,
                linewidth=2, markersize=5, label=label)
        ax.fill_between(sizes, gbps_min, gbps_max, color=color, alpha=0.15)

    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('relu_benchmark.png', dpi=150)
    print("\nPlot saved to relu_benchmark.png")
    plt.show()
