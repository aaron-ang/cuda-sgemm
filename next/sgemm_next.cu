// The "What we'd do next" steps from the write-up, applied one at a time on
// top of the course kernel, and benchmarked against cuBLAS.
//
//   k0  baseline      the course kernel (128x128 block tile, 8x8 per thread, BK=32)
//   k1  +launch       k0 + __launch_bounds__(256, 2): cap registers so 2 blocks fit per SM
//   k2  +float4       k1 + 128-bit global loads, A stored transposed in shared memory,
//                     128-bit shared-memory reads (nvcc already merges k0's reads
//                     along k into 128-bit loads; ncu shows equal shared-load counts)
//   k3  +banks        k2 + each thread's 8x8 outputs split into 4x4 quadrants 64 apart,
//                     so a warp's shared-memory reads are conflict-free
//   k4  +warptile     k3 + a 32x64 tile per warp, so a warp reads less shared memory per FMA
//   k5  +dbuf         k4 with two shared-memory buffers: the next K slice is fetched
//                     while the current one is computed (BK=16 so both buffers fit)
//
// Matrices are square, row-major, with a leading dimension ld = N rounded up to
// a multiple of 4 (zero padded) so every row starts 16-byte aligned. Any N works.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CK(x)                                                                  \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(e_));                                   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

constexpr int BM = 128, BN = 128, TM = 8, TN = 8, NT = 256;

// ---------------------------------------------------------------- k0 / k1

// Same algorithm and indexing as kernel/mmpy_kernel.cu, with a separate
// leading dimension and static shared memory.
__device__ __forceinline__ void baseline_body(int N, int ld, float *__restrict__ C,
                                              const float *__restrict__ A,
                                              const float *__restrict__ B)
{
    constexpr int BK = 32;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const int rowOff = blockIdx.y * BM, colOff = blockIdx.x * BN;
    A += (size_t)rowOff * ld;
    B += colOff;
    C += (size_t)rowOff * ld + colOff;

    const int tx = threadIdx.x, ty = threadIdx.y, t = ty * blockDim.x + tx;
    const int iRA = t / BK, iCA = t % BK, iRB = t / BN, iCB = t % BN;
    constexpr int sA = NT / BK, sB = NT / BN;

    float acc[TM][TN] = {};
    float rM[TM], rN[TN];

    for (int bk = 0; bk < N; bk += BK) {
#pragma unroll
        for (int o = 0; o < BM; o += sA) {
            int gr = rowOff + iRA + o, gc = bk + iCA;
            As[(iRA + o) * BK + iCA] = (gr < N && gc < N) ? A[(size_t)(iRA + o) * ld + iCA] : 0.f;
        }
#pragma unroll
        for (int o = 0; o < BK; o += sB) {
            int gr = bk + iRB + o, gc = colOff + iCB;
            Bs[(iRB + o) * BN + iCB] = (gr < N && gc < N) ? B[(size_t)(iRB + o) * ld + iCB] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int i = 0; i < TM; ++i) rM[i] = As[(ty * TM + i) * BK + k];
#pragma unroll
            for (int j = 0; j < TN; ++j) rN[j] = Bs[k * BN + tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] += rM[i] * rN[j];
        }
        __syncthreads();
        A += BK;
        B += (size_t)BK * ld;
    }

#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = ty * TM + i, c = tx * TN + j;
            if (rowOff + r < N && colOff + c < N) C[(size_t)r * ld + c] = acc[i][j];
        }
}

__global__ void k0_baseline(int N, int ld, float *C, const float *A, const float *B)
{
    baseline_body(N, ld, C, A, B);
}

__global__ void __launch_bounds__(NT, 2)
k1_launch(int N, int ld, float *C, const float *A, const float *B)
{
    baseline_body(N, ld, C, A, B);
}

// ---------------------------------------------------------------- tile loads

// Global -> registers -> shared memory, 128 bits at a time. Split in two so the
// double-buffered kernel can issue the global loads early.
template <int BK> struct Tiles {
    static constexpr int A4 = BK / 4, RA = NT / A4, PA = BM / RA;  // A: BM x BK
    static constexpr int B4 = BN / 4, RB = NT / B4, PB = BK / RB;  // B: BK x BN
    float4 a[PA], b[PB];

    __device__ __forceinline__ static float4 ld4(const float *p, bool ok)
    {
        return ok ? *reinterpret_cast<const float4 *>(p) : make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // Columns in [N, ld) are zero padding, so a float4 whose first column is
    // < N never reads past the row.
    __device__ __forceinline__ void fetch(int N, int ld, const float *A, const float *B,
                                          int rowOff, int colOff, int bk, int t)
    {
#pragma unroll
        for (int p = 0; p < PA; ++p) {
            int r = t / A4 + p * RA, c = (t % A4) * 4;
            a[p] = ld4(A + (size_t)(rowOff + r) * ld + bk + c, rowOff + r < N && bk + c < N);
        }
#pragma unroll
        for (int p = 0; p < PB; ++p) {
            int r = t / B4 + p * RB, c = (t % B4) * 4;
            b[p] = ld4(B + (size_t)(bk + r) * ld + colOff + c, bk + r < N && colOff + c < N);
        }
    }

    // A is stored transposed (As[k][m]) so a thread's row values are contiguous.
    __device__ __forceinline__ void stash(float *As, float *Bs, int t) const
    {
#pragma unroll
        for (int p = 0; p < PA; ++p) {
            int r = t / A4 + p * RA, c = (t % A4) * 4;
            As[(c + 0) * BM + r] = a[p].x;
            As[(c + 1) * BM + r] = a[p].y;
            As[(c + 2) * BM + r] = a[p].z;
            As[(c + 3) * BM + r] = a[p].w;
        }
#pragma unroll
        for (int p = 0; p < PB; ++p) {
            int r = t / B4 + p * RB, c = (t % B4) * 4;
            *reinterpret_cast<float4 *>(&Bs[r * BN + c]) = b[p];
        }
    }
};

// ---------------------------------------------------------------- thread layouts

// Where a thread's 8 rows (or columns) sit inside the 128x128 block tile. Each
// layout returns two 4-wide groups; the thread reads each group as one float4.
struct Contig {   // k2: 8 consecutive rows/cols, as in the course kernel
    __device__ static int row(int t, int g) { return (t / 16) * 8 + g * 4; }
    __device__ static int col(int t, int g) { return (t % 16) * 8 + g * 4; }
};
struct Split {    // k3: two groups of 4, 64 apart
    __device__ static int row(int t, int g) { return (t / 16) * 4 + g * 64; }
    __device__ static int col(int t, int g) { return (t % 16) * 4 + g * 64; }
};
struct Warp {     // k4/k5: 8 warps in a 4x2 grid of 32x64 warp tiles
    __device__ static int row(int t, int g)
    {
        int w = t / 32, lane = t % 32;
        return (w / 2) * 32 + (lane / 8) * 4 + g * 16;
    }
    __device__ static int col(int t, int g)
    {
        int w = t / 32, lane = t % 32;
        return (w % 2) * 64 + (lane % 8) * 4 + g * 32;
    }
};

template <int BK, class L>
__device__ __forceinline__ void compute_tile(const float *As, const float *Bs, int t,
                                             float (&acc)[TM][TN])
{
    const int r0 = L::row(t, 0), r1 = L::row(t, 1), c0 = L::col(t, 0), c1 = L::col(t, 1);
#pragma unroll
    for (int k = 0; k < BK; ++k) {
        float4 a0 = *reinterpret_cast<const float4 *>(&As[k * BM + r0]);
        float4 a1 = *reinterpret_cast<const float4 *>(&As[k * BM + r1]);
        float4 b0 = *reinterpret_cast<const float4 *>(&Bs[k * BN + c0]);
        float4 b1 = *reinterpret_cast<const float4 *>(&Bs[k * BN + c1]);
        const float rM[TM] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const float rN[TN] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] += rM[i] * rN[j];
    }
}

template <class L>
__device__ __forceinline__ void store_tile(int N, int ld, float *C, int rowOff, int colOff,
                                           int t, const float (&acc)[TM][TN])
{
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = rowOff + L::row(t, i / 4) + i % 4;
        if (r >= N) continue;
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            int c = colOff + L::col(t, g);
            if (c < N)   // c is a multiple of 4 and ld >= N rounded up to 4
                *reinterpret_cast<float4 *>(&C[(size_t)r * ld + c]) =
                    make_float4(acc[i][g * 4], acc[i][g * 4 + 1], acc[i][g * 4 + 2], acc[i][g * 4 + 3]);
        }
    }
}

// ---------------------------------------------------------------- k2 .. k4

template <int BK, class L>
__global__ void __launch_bounds__(NT, 2)
k_single(int N, int ld, float *C, const float *A, const float *B)
{
    __shared__ __align__(16) float As[BK * BM];
    __shared__ __align__(16) float Bs[BK * BN];
    const int t = threadIdx.x, rowOff = blockIdx.y * BM, colOff = blockIdx.x * BN;
    float acc[TM][TN] = {};
    Tiles<BK> tiles;

    for (int bk = 0; bk < N; bk += BK) {
        tiles.fetch(N, ld, A, B, rowOff, colOff, bk, t);
        tiles.stash(As, Bs, t);
        __syncthreads();
        compute_tile<BK, L>(As, Bs, t, acc);
        __syncthreads();
    }
    store_tile<L>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- k5

template <int BK, class L>
__global__ void __launch_bounds__(NT, 2)
k_dbuf(int N, int ld, float *C, const float *A, const float *B)
{
    __shared__ __align__(16) float As[2][BK * BM];
    __shared__ __align__(16) float Bs[2][BK * BN];
    const int t = threadIdx.x, rowOff = blockIdx.y * BM, colOff = blockIdx.x * BN;
    float acc[TM][TN] = {};
    Tiles<BK> tiles;

    tiles.fetch(N, ld, A, B, rowOff, colOff, 0, t);
    tiles.stash(As[0], Bs[0], t);
    __syncthreads();

    int cur = 0;
    for (int bk = 0; bk < N; bk += BK) {
        const bool more = bk + BK < N;
        if (more)   // global loads in flight while we compute
            tiles.fetch(N, ld, A, B, rowOff, colOff, bk + BK, t);
        compute_tile<BK, L>(As[cur], Bs[cur], t, acc);
        if (more)   // the other buffer was last read before the previous barrier
            tiles.stash(As[cur ^ 1], Bs[cur ^ 1], t);
        __syncthreads();
        cur ^= 1;
    }
    store_tile<L>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- harness

typedef void (*Kernel)(int, int, float *, const float *, const float *);

struct Variant {
    const char *name;
    Kernel fn;
    bool block2d;   // k0/k1 use the course's 16x16 thread block
};

static const Variant variants[] = {
    {"k0 baseline", k0_baseline, true},
    {"k1 +launch_bounds", k1_launch, true},
    {"k2 +float4", k_single<32, Contig>, false},
    {"k3 +bank-conflict fix", k_single<32, Split>, false},
    {"k4 +warptile", k_single<32, Warp>, false},
    {"k4 +warptile (BK=16)", k_single<16, Warp>, false},
    {"k5 +double buffer (BK=16)", k_dbuf<16, Warp>, false},
};

static void launch(const Variant &v, int n, int ld, float *C, const float *A, const float *B)
{
    dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    dim3 block = v.block2d ? dim3(16, 16) : dim3(NT);
    v.fn<<<grid, block>>>(n, ld, C, A, B);
}

// Median GFLOP/s over `trials`, each timing `reps` back-to-back calls.
template <class F>
static void time_it(F f, int n, double &med, double &lo, double &hi)
{
    const double flop = 2.0 * n * (double)n * n;
    const int reps = std::max(10, std::min(2000, (int)std::ceil(4e11 / flop)));
    const int trials = 7;
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; ++i) f();
    std::vector<double> g;
    for (int tr = 0; tr < trials; ++tr) {
        CK(cudaEventRecord(e0));
        for (int r = 0; r < reps; ++r) f();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        g.push_back(flop * reps / (ms * 1e-3) / 1e9);
    }
    CK(cudaGetLastError());
    std::sort(g.begin(), g.end());
    med = g[trials / 2];
    lo = g.front();
    hi = g.back();
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

int main(int argc, char **argv)
{
    setenv("NVIDIA_TF32_OVERRIDE", "0", 1);   // keep cuBLAS in true FP32
    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
    if (sizes.empty()) sizes = {1024, 1025, 2048, 2049, 4096};

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("# %s, %d SMs, CC %d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    for (const Variant &v : variants) {
        cudaFuncAttributes a;
        CK(cudaFuncGetAttributes(&a, v.fn));
        int blocks;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, v.fn, NT, 0));
        printf("# %-28s regs=%3d smem=%5zu B  blocks/SM=%d  spill_ld=%zu B\n", v.name,
               a.numRegs, a.sharedSizeBytes, blocks, a.localSizeBytes);
    }

    cublasHandle_t h;
    cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

    printf("%-28s %6s %10s %10s %10s %9s %9s\n", "kernel", "N", "GFLOP/s", "min", "max",
           "%cuBLAS", "max_err");
    for (int n : sizes) {
        const int ld = (n + 3) & ~3;
        const size_t elems = (size_t)n * ld;
        std::vector<float> hA(elems, 0.f), hB(elems, 0.f), hC(elems), hR(elems);
        srand(n);
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) {
                hA[(size_t)r * ld + c] = 2.f * rand() / RAND_MAX - 1.f;
                hB[(size_t)r * ld + c] = 2.f * rand() / RAND_MAX - 1.f;
            }
        float *A, *B, *C, *R;
        CK(cudaMalloc(&A, elems * 4));
        CK(cudaMalloc(&B, elems * 4));
        CK(cudaMalloc(&C, elems * 4));
        CK(cudaMalloc(&R, elems * 4));
        CK(cudaMemcpy(A, hA.data(), elems * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(B, hB.data(), elems * 4, cudaMemcpyHostToDevice));

        // Row-major C = A B is column-major C^T = B^T A^T.
        const float one = 1.f, zero = 0.f;
        auto ref = [&] { cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, B, ld, A, ld, &zero, R, ld); };
        ref();
        CK(cudaMemcpy(hR.data(), R, elems * 4, cudaMemcpyDeviceToHost));
        float scale = 0.f;
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c) scale = std::max(scale, std::fabs(hR[(size_t)r * ld + c]));

        // Spot-check the cuBLAS reference itself against a double-precision dot product.
        double ref_err = 0;
        for (int s = 0; s < 64; ++s) {
            int r = rand() % n, c = rand() % n;
            double d = 0;
            for (int k = 0; k < n; ++k) d += (double)hA[(size_t)r * ld + k] * hB[(size_t)k * ld + c];
            ref_err = std::max(ref_err, std::fabs(d - hR[(size_t)r * ld + c]) / scale);
        }
        if (ref_err > 1e-4) {
            fprintf(stderr, "cuBLAS reference off by %.1e at N=%d\n", ref_err, n);
            exit(1);
        }

        // SGEMM_PROFILE=1: launch every kernel once and skip timing, for ncu.
        const bool profile = getenv("SGEMM_PROFILE") != nullptr;
        double cmed = 0, clo = 0, chi = 0;
        if (!profile)
            time_it(ref, n, cmed, clo, chi);

        for (const Variant &v : variants) {
            CK(cudaMemset(C, 0, elems * 4));
            launch(v, n, ld, C, A, B);
            CK(cudaGetLastError());
            CK(cudaMemcpy(hC.data(), C, elems * 4, cudaMemcpyDeviceToHost));
            float err = 0.f;
            for (int r = 0; r < n; ++r)
                for (int c = 0; c < n; ++c)
                    err = std::max(err, std::fabs(hC[(size_t)r * ld + c] - hR[(size_t)r * ld + c]));
            err /= scale;
            double med = 0, lo = 0, hi = 0;
            if (!profile)
                time_it([&] { launch(v, n, ld, C, A, B); }, n, med, lo, hi);
            printf("%-28s %6d %10.1f %10.1f %10.1f %8.1f%% %9.1e%s\n", v.name, n, med, lo, hi,
                   100 * med / cmed, err, err > 1e-4f ? "  WRONG" : "");
        }
        printf("%-28s %6d %10.1f %10.1f %10.1f %8.1f%% %9.1e (vs FP64, max |C|=%.1f)\n", "cuBLAS", n,
               cmed, clo, chi, 100.0, ref_err, scale);
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(R);
    }
    cublasDestroy(h);
    return 0;
}
