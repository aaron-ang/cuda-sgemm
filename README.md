# cuda-sgemm

A high-performance single-precision matrix-multiply (SGEMM) CUDA kernel, tuned for
the NVIDIA T4. It reaches **4779.9 GFLOP/s at N = 2048** — approaching cuBLAS —
through global-memory coalescing, shared-memory tiling, and two-dimensional register
blocking.

Developed and benchmarked on an NVIDIA T4. Full write-up, including the roofline and
occupancy analysis: **[docs/report.pdf](docs/report.pdf)** (LaTeX source in
[`docs/report/`](docs/report/)).

## Approach

- **Two-dimensional block tiling.** 16×16 thread blocks, each computing a 128×128
  output tile of C (an 8×8 micro-tile per thread) over 32-deep K sub-tiles staged in
  shared memory.
- **Coalesced global loads + shared-memory staging**, with bounds checking so
  matrix sizes that are not multiples of 128 stay correct.
- **Register-blocked outer products** with `#pragma unroll` to expose
  instruction-level parallelism.
- **Roofline / occupancy analysis** of register pressure and arithmetic intensity to
  explain where the kernel is compute- vs. memory-bound (see the report).
- Bonus: a **non-square** matrix-multiply kernel.

## Results (NVIDIA T4)

| N | Peak GFLOP/s | Thread-block config |
|---|---|---|
| 256 | 1148.9 | `bx=16 by=16 tm=4 tn=4` |
| 512 | 2166.9 | `bx=16 by=16 tm=8 tn=8` |
| 1024 | 3919.6 | `bx=16 by=16 tm=8 tn=8` |
| 2048 | 4779.9 | `bx=16 by=16 tm=8 tn=8` |

![Roofline model at N=2048](docs/roofline.png)

## Round two: the follow-ups, on a GB10

[`next/sgemm_next.cu`](next/sgemm_next.cu) applies the report's "next steps" one
at a time on top of the course kernel and benchmarks each against cuBLAS (FP32,
TF32 off). Measured on an NVIDIA GB10 (Blackwell, 48 SMs), so the numbers
are **not comparable** with the T4 results above.

| Step | N=2048 GFLOP/s | vs cuBLAS | N=4096 GFLOP/s | vs cuBLAS |
|---|---|---|---|---|
| Course kernel | 9,673 | 58% | 10,540 | 64% |
| + `__launch_bounds__(256,2)` | 11,266 | 68% | 12,173 | 74% |
| + `float4` loads, A transposed in smem | 13,439 | 81% | 13,692 | 83% |
| + conflict-free smem reads | 13,846 | 83% | 14,060 | 85% |
| + warp tiling (BK=32 / BK=16) | 13,809 / 14,261 | 83% / 86% | 14,268 / 14,786 | 86% / 89% |
| + double buffering (BK=16) | 14,600 | 88% | 15,010 | 91% |
| cuBLAS | 16,603 | 100% | 16,545 | 100% |

```bash
cd next
nvcc -O3 -std=c++17 -arch=sm_121 -o sgemm_next sgemm_next.cu -lcublas   # use your GPU's sm_XX
./sgemm_next 1024 1025 2048 2049 4096
```

## Layout

| Path | Contents |
|---|---|
| `kernel/` | **The core of the project** — the tuned CUDA kernel (`mmpy_kernel.cu`), grid setup (`setGrid.cu`), and tuning parameters (`OPTIONS.txt`). |
| `src/` | Benchmarking / host harness (provided by the course). |
| `build/` | Makefile and CUDA build configuration. |
| `tools/` | Profiling scripts (`run_ncu.sh`, `run_nvprof.sh`) and roofline plotting (`plot_roofline.py`). |
| `docs/` | Report and roofline figures. |

## Build & run

Requires a CUDA toolkit and an NVIDIA GPU (developed and benchmarked on a T4). Build
flags are read from `kernel/OPTIONS.txt`.

```bash
# build
make -C build `cat kernel/OPTIONS.txt`

# build with a cuBLAS reference for comparison
make -C build cublastest=1 `cat kernel/OPTIONS.txt`

# run at a given size
./mmpy `cat kernel/OPTIONS_RUNTIME.txt` -n 2048

# clean
make -C build clean
```

## Authors

Aaron Ang and Jerry Ma. The `src/` benchmarking harness is provided starter code; the
CUDA kernel and tuning in `kernel/` are our work.
