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
