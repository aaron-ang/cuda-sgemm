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
// Round three, closing the gap to cuBLAS. k6, k7 and k9 didn't help and are kept
// so the numbers can be reproduced:
//   k6  +pairswap     k5 with B's column pairs swapped in shared memory, to move FFMA
//                     operands out of the same register bank (no gain)
//   k7  +cp.async     k5's tiling, loaded with cp.async into a 2- or 4-stage pipeline
//                     (no gain: loads were already hidden)
//   k8  +bigtile      128x256 block tile, 256 threads with 16x8 outputs each, cp.async,
//                     BK=16, 2 stages, 1 block/SM
//   k9  +splitK       k8 with K split 3 ways, partial sums added with atomics
//                     (faster only when k8's grid leaves SMs idle)
//   k10 +kmajor       128x128 tile, 128 threads with 16x8 each, A kept k-major in shared
//                     memory so it's copied 16 bytes at a time (smaller tile for odd sizes)
//   k11 +pick         per size, runs whichever of k8/k9/k10 leaves the busiest SM the
//                     least work, as cuBLAS's heuristics do
//   k12 +kmajor2      k10's k-major A at 128x256 under a register cap (dropped: 2-4% below k8)
//   k13 +quad         k8 with each half-warp on 4x4 lanes, meant to cut shared-load
//                     wavefronts as cuBLAS does; ncu shows ~25% MORE (dropped, = k8)
//   k14 +splitwave    k8, but when the last wave is partial the first block rows are cut
//                     into 2-4 K pieces spread over all SMs; the last piece of a tile adds
//                     the others' sums from a workspace. Then whole-tile waves for the rest
//   k15 +pick2        k11 when there are fewer tiles than SMs, else k14
//
// SGEMM_ONLY=k5,k8 runs only kernels whose names start with those prefixes.
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

// SA is the row stride of the transposed A tile (k7+ pad it to BM + 4).
template <int BK, class L, int SA = BM>
__device__ __forceinline__ void compute_tile(const float *As, const float *Bs, int t,
                                             float (&acc)[TM][TN])
{
    const int r0 = L::row(t, 0), r1 = L::row(t, 1), c0 = L::col(t, 0), c1 = L::col(t, 1);
#pragma unroll
    for (int k = 0; k < BK; ++k) {
        float4 a0 = *reinterpret_cast<const float4 *>(&As[k * SA + r0]);
        float4 a1 = *reinterpret_cast<const float4 *>(&As[k * SA + r1]);
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

// ---------------------------------------------------------------- k6

// An FFMA reads three registers, and two operands in the same register bank
// can cost an extra cycle. In k5, C is stored as float4, so acc[i][j..j+3] sit
// in an aligned register quad, and so do the B values from a 128-bit shared
// read: acc[i][j] and b[j] share an offset within their quads. Storing B's tile
// with each pair of columns swapped (c+1, c, c+3, c+2) breaks that pairing.
// It measured slightly slower (and spilled more), so it's a dead end here.
template <int BK> struct TilesSwap : Tiles<BK> {
    using Tiles<BK>::a;
    using Tiles<BK>::b;
    using Tiles<BK>::A4;
    using Tiles<BK>::RA;
    using Tiles<BK>::PA;
    using Tiles<BK>::B4;
    using Tiles<BK>::RB;
    using Tiles<BK>::PB;

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
            *reinterpret_cast<float4 *>(&Bs[r * BN + c]) = make_float4(b[p].y, b[p].x, b[p].w, b[p].z);
        }
    }
};

template <int BK, class L>
__device__ __forceinline__ void compute_tile_swap(const float *As, const float *Bs, int t,
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
        const float rN[TN] = {b0.y, b0.x, b0.w, b0.z, b1.y, b1.x, b1.w, b1.z};
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] += rM[i] * rN[j];
    }
}

template <int BK, class L>
__global__ void __launch_bounds__(NT, 2)
k_swap(int N, int ld, float *C, const float *A, const float *B)
{
    __shared__ __align__(16) float As[2][BK * BM];
    __shared__ __align__(16) float Bs[2][BK * BN];
    const int t = threadIdx.x, rowOff = blockIdx.y * BM, colOff = blockIdx.x * BN;
    float acc[TM][TN] = {};
    TilesSwap<BK> tiles;

    tiles.fetch(N, ld, A, B, rowOff, colOff, 0, t);
    tiles.stash(As[0], Bs[0], t);
    __syncthreads();

    int cur = 0;
    for (int bk = 0; bk < N; bk += BK) {
        const bool more = bk + BK < N;
        if (more)
            tiles.fetch(N, ld, A, B, rowOff, colOff, bk + BK, t);
        compute_tile_swap<BK, L>(As[cur], Bs[cur], t, acc);
        if (more)
            tiles.stash(As[cur ^ 1], Bs[cur ^ 1], t);
        __syncthreads();
        cur ^= 1;
    }
    store_tile<L>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- k7

// cp.async copies global -> shared memory without passing through registers.
// src_bytes = 0 zero-fills the destination (used for out-of-range elements).
__device__ __forceinline__ void cp_async4(float *dst, const float *src, bool ok)
{
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 4 : 0));
}
__device__ __forceinline__ void cp_async16(float *dst, const float *src, bool ok)
{
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// One K slice of A (transposed, row stride SA = BM + 4) and B, copied with cp.async.
// A can't be transposed by a 16-byte copy, so it goes 4 bytes at a time: each
// warp instruction covers 8 consecutive k (one 32-byte sector per row) of 4
// rows. With the +4 padding, those 32 floats land in 32 different banks.
template <int BK> struct AsyncTiles {
    static constexpr int SA = BM + 4;
    static constexpr int A_STAGE = BK * SA, B_STAGE = BK * BN, STAGE_BYTES = (A_STAGE + B_STAGE) * 4;
    static constexpr int QA = BK / 8 * BM / 4;           // warp instructions per A slice
    static constexpr int PA = QA / (NT / 32);             // ... per warp
    static constexpr int B4 = BN / 4, RB = NT / B4, PB = BK / RB;

    __device__ __forceinline__ static void load(int N, int ld, const float *A, const float *B,
                                                float *As, float *Bs, int rowOff, int colOff,
                                                int bk, int t)
    {
        const int w = t / 32, lane = t % 32;
#pragma unroll
        for (int p = 0; p < PA; ++p) {
            int q = w * PA + p;
            int k = (q % (BK / 8)) * 8 + lane % 8, m = (q / (BK / 8)) * 4 + lane / 8;
            bool ok = rowOff + m < N && bk + k < N;
            cp_async4(&As[k * SA + m], ok ? A + (size_t)(rowOff + m) * ld + bk + k : A, ok);
        }
#pragma unroll
        for (int p = 0; p < PB; ++p) {
            int r = t / B4 + p * RB, c = (t % B4) * 4;
            bool ok = bk + r < N && colOff + c < N;
            cp_async16(&Bs[r * BN + c], ok ? B + (size_t)(bk + r) * ld + colOff + c : B, ok);
        }
    }
};

// S-stage pipeline: S-1 slices are in flight while one is computed.
template <int BK, int S, class L>
__global__ void __launch_bounds__(NT, 2)
k_async(int N, int ld, float *C, const float *A, const float *B)
{
    using T = AsyncTiles<BK>;
    extern __shared__ __align__(16) float smem[];
    float *As = smem, *Bs = smem + S * T::A_STAGE;
    const int t = threadIdx.x, rowOff = blockIdx.y * BM, colOff = blockIdx.x * BN;
    const int KT = (N + BK - 1) / BK;
    float acc[TM][TN] = {};

#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KT)
            T::load(N, ld, A, B, As + s * T::A_STAGE, Bs + s * T::B_STAGE, rowOff, colOff, s * BK, t);
        cp_async_commit();
    }
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();   // slice kt has landed (for this thread) ...
        __syncthreads();          // ... and for every thread; slot (kt-1)%S is free
        const int nk = kt + S - 1;
        if (nk < KT)
            T::load(N, ld, A, B, As + (nk % S) * T::A_STAGE, Bs + (nk % S) * T::B_STAGE, rowOff,
                    colOff, nk * BK, t);
        cp_async_commit();
        compute_tile<BK, L, T::SA>(As + (kt % S) * T::A_STAGE, Bs + (kt % S) * T::B_STAGE, t, acc);
    }
    store_tile<L>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- k8 / k9

// Bigger per-thread tiles, fewer threads per output. cuBLAS runs
// cutlass_80_simt_sgemm_128x128_8x4 here: 128 threads with 128 outputs each.
// A thread doing TR x TC FMAs per k reads (TR + TC) / 4 float4s from shared
// memory, so 16x8 needs a quarter less shared traffic per FMA than 8x8, and
// fewer instructions that aren't FMAs. Shape bundles the tiling:
//   block tile BM x BN, K slice BK, S cp.async stages, NTH threads,
//   warp tile WM x WN with its lanes arranged LR x LC, MINB blocks per SM.
template <int BM_, int BN_, int BK_, int S_, int NTH_, int WM_, int WN_, int LR_, int MINB_>
struct Shape {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, S = S_, NTH = NTH_, MINB = MINB_;
    static constexpr int WM = WM_, WN = WN_, LR = LR_, LC = 32 / LR_;
    static constexpr int TR = WM / LR, TC = WN / LC;     // thread tile
    static constexpr int WGC = BN / WN;                  // warps across a block row
    static constexpr int SA = BM + 4;                    // padded stride of transposed A
    static constexpr int A_STAGE = BK * SA, B_STAGE = BK * BN;
    static constexpr int SMEM = S * (A_STAGE + B_STAGE) * 4;
    static_assert((BM / WM) * (BN / WN) * 32 == NTH, "warps must cover the block tile");
    static_assert(TR % 4 == 0 && TC % 4 == 0 && BK % 8 == 0, "shape");
    // first row/column of the thread's g-th group of 4
    __device__ static int row(int t, int g) { return (t / 32 / WGC) * WM + g * LR * 4 + (t % 32 / LC) * 4; }
    __device__ static int col(int t, int g) { return (t / 32 % WGC) * WN + g * LC * 4 + (t % 32 % LC) * 4; }
};

// Same copy scheme as AsyncTiles, for any Shape. Rows/columns of C must be
// < N, k must be < K.
template <class P>
__device__ __forceinline__ void load_slice(int N, int K, int ld, const float *A, const float *B,
                                           float *As, float *Bs, int rowOff, int colOff,
                                           int bk, int t)
{
    constexpr int QA = P::BK / 8 * P::BM / 4, PA = QA / (P::NTH / 32);
    static_assert(QA % (P::NTH / 32) == 0, "A copy split");
    const int w = t / 32, lane = t % 32;
#pragma unroll
    for (int p = 0; p < PA; ++p) {
        int q = w * PA + p;
        int k = (q % (P::BK / 8)) * 8 + lane % 8, m = (q / (P::BK / 8)) * 4 + lane / 8;
        bool ok = rowOff + m < N && bk + k < K;
        cp_async4(&As[k * P::SA + m], ok ? A + (size_t)(rowOff + m) * ld + bk + k : A, ok);
    }
    constexpr int B4 = P::BN / 4, RB = P::NTH / B4, PB = P::BK / RB;
    static_assert(P::NTH % B4 == 0 && P::BK % RB == 0, "B copy split");
#pragma unroll
    for (int p = 0; p < PB; ++p) {
        int r = t / B4 + p * RB, c = (t % B4) * 4;
        bool ok = bk + r < K && colOff + c < N;
        cp_async16(&Bs[r * P::BN + c], ok ? B + (size_t)(bk + r) * ld + colOff + c : B, ok);
    }
}

// Store a thread's TR x TC outputs; SPLIT adds them to C with 128-bit atomics.
template <class P, bool SPLIT, int TR, int TC>
__device__ __forceinline__ void store_big(int N, int ld, float *C, int rowOff, int colOff, int t,
                                          const float (&acc)[TR][TC])
{
#pragma unroll
    for (int i = 0; i < TR; ++i) {
        int r = rowOff + P::row(t, i / 4) + i % 4;
        if (r >= N) continue;
#pragma unroll
        for (int g = 0; g < TC / 4; ++g) {
            int c = colOff + P::col(t, g);
            if (c >= N) continue;
            float4 v = make_float4(acc[i][g * 4], acc[i][g * 4 + 1], acc[i][g * 4 + 2], acc[i][g * 4 + 3]);
            if (SPLIT)
                atomicAdd(reinterpret_cast<float4 *>(&C[(size_t)r * ld + c]), v);
            else
                *reinterpret_cast<float4 *>(&C[(size_t)r * ld + c]) = v;
        }
    }
}

// SPLIT (k9): blockIdx.z takes a 1/gridDim.z share of the K slices and adds its
// partial sums into C, which the host zeroes first.
template <class P, bool SPLIT = false>
__global__ void __launch_bounds__(P::NTH, P::MINB)
k_big(int N, int ld, float *C, const float *A, const float *B)
{
    constexpr int BK = P::BK, S = P::S, TR = P::TR, TC = P::TC;
    extern __shared__ __align__(16) float smem[];
    float *As = smem, *Bs = smem + S * P::A_STAGE;
    const int t = threadIdx.x, rowOff = blockIdx.y * P::BM, colOff = blockIdx.x * P::BN;
    int K = N, KT = (N + BK - 1) / BK;
    if (SPLIT) {   // this block's K slices
        const int per = (KT + gridDim.z - 1) / gridDim.z, k0 = blockIdx.z * per;
        KT = max(0, min(KT - k0, per));
        A += k0 * BK;
        B += (size_t)k0 * BK * ld;
        K -= k0 * BK;
    }
    float acc[TR][TC] = {};

#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KT)
            load_slice<P>(N, K, ld, A, B, As + s * P::A_STAGE, Bs + s * P::B_STAGE, rowOff, colOff, s * BK, t);
        cp_async_commit();
    }
    const int r0 = P::row(t, 0), c0 = P::col(t, 0);
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        const int nk = kt + S - 1;
        if (nk < KT)
            load_slice<P>(N, K, ld, A, B, As + (nk % S) * P::A_STAGE, Bs + (nk % S) * P::B_STAGE,
                          rowOff, colOff, nk * BK, t);
        cp_async_commit();
        const float *as = As + (kt % S) * P::A_STAGE + r0, *bs = Bs + (kt % S) * P::B_STAGE + c0;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rM[TR], rN[TC];
#pragma unroll
            for (int g = 0; g < TR / 4; ++g)
                *reinterpret_cast<float4 *>(&rM[g * 4]) =
                    *reinterpret_cast<const float4 *>(&as[k * P::SA + g * P::LR * 4]);
#pragma unroll
            for (int g = 0; g < TC / 4; ++g)
                *reinterpret_cast<float4 *>(&rN[g * 4]) =
                    *reinterpret_cast<const float4 *>(&bs[k * P::BN + g * P::LC * 4]);
#pragma unroll
            for (int i = 0; i < TR; ++i)
#pragma unroll
                for (int j = 0; j < TC; ++j) acc[i][j] += rM[i] * rN[j];
        }
    }
    store_big<P, SPLIT>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- k10

// A kept k-major in shared memory (As[m][k], as in global memory), so it can be
// copied 16 bytes at a time like B instead of 4. A thread then reads A along k:
// one 128-bit read gives 4 k values of one row, used for 4 x TC FMAs. Rows are
// BK floats long; the 16-byte chunks of row m are XOR-swizzled by (m / 4) % 4 so
// the 4 rows a warp reads at once (4 apart) hit different banks.
template <class P> struct KMajor {
    static constexpr int BK = P::BK, A_STAGE = P::BM * BK, B_STAGE = BK * P::BN;
    static constexpr int SMEM = P::S * (A_STAGE + B_STAGE) * 4;
    static_assert(BK == 16, "swizzle assumes 4 chunks per row");
    __device__ static int aoff(int m, int kq) { return m * BK + ((kq ^ (m / 4)) & 3) * 4; }

    __device__ __forceinline__ static void load(int N, int ld, const float *A, const float *B,
                                                float *As, float *Bs, int rowOff, int colOff,
                                                int bk, int t)
    {
        constexpr int A4 = BK / 4, RA = P::NTH / A4, PA = P::BM / RA;
#pragma unroll
        for (int p = 0; p < PA; ++p) {
            int m = t / A4 + p * RA, kq = t % A4;
            bool ok = rowOff + m < N && bk + kq * 4 < N;   // padding keeps the float4 in the row
            cp_async16(&As[aoff(m, kq)], ok ? A + (size_t)(rowOff + m) * ld + bk + kq * 4 : A, ok);
        }
        constexpr int B4 = P::BN / 4, RB = P::NTH / B4, PB = BK / RB;
#pragma unroll
        for (int p = 0; p < PB; ++p) {
            int r = t / B4 + p * RB, c = (t % B4) * 4;
            bool ok = bk + r < N && colOff + c < N;
            cp_async16(&Bs[r * P::BN + c], ok ? B + (size_t)(bk + r) * ld + colOff + c : B, ok);
        }
    }
};

template <class P>
__global__ void __launch_bounds__(P::NTH, P::MINB)
k_kmajor(int N, int ld, float *C, const float *A, const float *B)
{
    using T = KMajor<P>;
    constexpr int BK = P::BK, S = P::S, TR = P::TR, TC = P::TC;
    extern __shared__ __align__(16) float smem[];
    float *As = smem, *Bs = smem + S * T::A_STAGE;
    const int t = threadIdx.x, rowOff = blockIdx.y * P::BM, colOff = blockIdx.x * P::BN;
    const int KT = (N + BK - 1) / BK;
    float acc[TR][TC] = {};

#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KT)
            T::load(N, ld, A, B, As + s * T::A_STAGE, Bs + s * T::B_STAGE, rowOff, colOff, s * BK, t);
        cp_async_commit();
    }
    const int c0 = P::col(t, 0);
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        const int nk = kt + S - 1;
        if (nk < KT)
            T::load(N, ld, A, B, As + (nk % S) * T::A_STAGE, Bs + (nk % S) * T::B_STAGE, rowOff, colOff,
                    nk * BK, t);
        cp_async_commit();
        const float *as = As + (kt % S) * T::A_STAGE, *bs = Bs + (kt % S) * T::B_STAGE + c0;
#pragma unroll
        for (int kq = 0; kq < BK / 4; ++kq) {
            float rN[4][TC];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int g = 0; g < TC / 4; ++g)
                    *reinterpret_cast<float4 *>(&rN[kk][g * 4]) =
                        *reinterpret_cast<const float4 *>(&bs[(kq * 4 + kk) * P::BN + g * P::LC * 4]);
#pragma unroll
            for (int i = 0; i < TR; ++i) {
                const int m = P::row(t, i / 4) + i % 4;
                const float4 a = *reinterpret_cast<const float4 *>(&as[T::aoff(m, kq)]);
                const float ak[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                    for (int j = 0; j < TC; ++j) acc[i][j] += ak[kk] * rN[kk][j];
            }
        }
    }
    store_big<P, false>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- k12

// k10's k-major A, trimmed for registers so it fits the 128x256 tile. All of a
// thread's rows share one swizzle key ((row / 4) % 4 doesn't change across its
// row groups when LR is a multiple of 4), so the four chunk offsets are
// computed once instead of per row. MAXR caps registers (__maxnreg__).
template <class P, int MAXR = 0>
__global__ void __launch_bounds__(P::NTH) __maxnreg__(MAXR ? MAXR : 65536 / (P::NTH * P::MINB) / 8 * 8 > 255 ? 255 : 65536 / (P::NTH * P::MINB) / 8 * 8)
k_kmajor2(int N, int ld, float *C, const float *A, const float *B)
{
    using T = KMajor<P>;
    constexpr int BK = P::BK, S = P::S, TR = P::TR, TC = P::TC;
    static_assert(P::LR % 4 == 0, "one swizzle key per thread");
    extern __shared__ __align__(16) float smem[];
    float *As = smem, *Bs = smem + S * T::A_STAGE;
    const int t = threadIdx.x, rowOff = blockIdx.y * P::BM, colOff = blockIdx.x * P::BN;
    const int KT = (N + BK - 1) / BK;
    float acc[TR][TC] = {};

#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (s < KT)
            T::load(N, ld, A, B, As + s * T::A_STAGE, Bs + s * T::B_STAGE, rowOff, colOff, s * BK, t);
        cp_async_commit();
    }
    const int r0 = P::row(t, 0), c0 = P::col(t, 0), key = r0 / 4;
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        const int nk = kt + S - 1;
        if (nk < KT)
            T::load(N, ld, A, B, As + (nk % S) * T::A_STAGE, Bs + (nk % S) * T::B_STAGE, rowOff, colOff,
                    nk * BK, t);
        cp_async_commit();
        const float *as = As + (kt % S) * T::A_STAGE + r0 * BK, *bs = Bs + (kt % S) * T::B_STAGE + c0;
#pragma unroll
        for (int kq = 0; kq < BK / 4; ++kq) {
            const float *ak = as + ((kq ^ key) & 3) * 4;
            float rN[4][TC];
#pragma unroll
            for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                for (int g = 0; g < TC / 4; ++g)
                    *reinterpret_cast<float4 *>(&rN[kk][g * 4]) =
                        *reinterpret_cast<const float4 *>(&bs[(kq * 4 + kk) * P::BN + g * P::LC * 4]);
#pragma unroll
            for (int i = 0; i < TR; ++i) {
                const int m = (i / 4) * P::LR * 4 + i % 4;   // row offset from r0
                const float4 a = *reinterpret_cast<const float4 *>(&ak[m * BK]);
                const float av[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
                for (int kk = 0; kk < 4; ++kk)
#pragma unroll
                    for (int j = 0; j < TC; ++j) acc[i][j] += av[kk] * rN[kk][j];
            }
        }
    }
    store_big<P, false>(N, ld, C, rowOff, colOff, t, acc);
}

// ---------------------------------------------------------------- k13

// k8 with the lanes of each warp rearranged. ncu shows a 128-bit shared load
// costs one wavefront per 64 distinct bytes in each half-warp (min 1). In k8,
// a half-warp covers 2 rows x 8 columns, so its B loads touch 128 bytes (2
// wavefronts per half) and its A loads 32 (1). Arranging each half-warp as
// 4 rows x 4 columns (as cuBLAS does) makes both 64 bytes: one wavefront per
// half-warp for every load, a quarter fewer wavefronts in total for the same
// loads and FMAs.
// Measured (ncu, N=4096): the opposite. Same 101M LDS as k8, but 336M
// wavefronts vs k8's 268M (cuBLAS: 202M), so the model above is wrong.
template <class Base> struct Quad : Base {
    static_assert(Base::LR == 4 && Base::LC == 8, "4x8 lanes");
    __device__ static int row(int t, int g)
    {
        const int lane = t % 32;
        return (t / 32 / Base::WGC) * Base::WM + g * 16 + (lane % 4) * 4;
    }
    __device__ static int col(int t, int g)
    {
        const int lane = t % 32;
        return (t / 32 % Base::WGC) * Base::WN + g * 32 + (lane / 4 % 4 + lane / 16 * 4) * 4;
    }
};

// ---------------------------------------------------------------- k14

// k8's main loop over K slices [kb, ke) of one tile, accumulating into acc.
template <class P>
__device__ __forceinline__ void mainloop(int N, int ld, const float *A, const float *B, float *As,
                                         float *Bs, int rowOff, int colOff, int kb, int ke, int t,
                                         float (&acc)[P::TR][P::TC])
{
    constexpr int BK = P::BK, S = P::S, TR = P::TR, TC = P::TC;
    __syncthreads();   // the previous tile's last slice may still be being read
#pragma unroll
    for (int s = 0; s < S - 1; ++s) {
        if (kb + s < ke)
            load_slice<P>(N, N, ld, A, B, As + s * P::A_STAGE, Bs + s * P::B_STAGE, rowOff, colOff,
                          (kb + s) * BK, t);
        cp_async_commit();
    }
    const int r0 = P::row(t, 0), c0 = P::col(t, 0);
    for (int kt = 0; kt < ke - kb; ++kt) {
        cp_async_wait<S - 2>();
        __syncthreads();
        const int nk = kt + S - 1;
        if (kb + nk < ke)
            load_slice<P>(N, N, ld, A, B, As + (nk % S) * P::A_STAGE, Bs + (nk % S) * P::B_STAGE,
                          rowOff, colOff, (kb + nk) * BK, t);
        cp_async_commit();
        const float *as = As + (kt % S) * P::A_STAGE + r0, *bs = Bs + (kt % S) * P::B_STAGE + c0;
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float rM[TR], rN[TC];
#pragma unroll
            for (int g = 0; g < TR / 4; ++g)
                *reinterpret_cast<float4 *>(&rM[g * 4]) =
                    *reinterpret_cast<const float4 *>(&as[k * P::SA + g * P::LR * 4]);
#pragma unroll
            for (int g = 0; g < TC / 4; ++g)
                *reinterpret_cast<float4 *>(&rN[g * 4]) =
                    *reinterpret_cast<const float4 *>(&bs[k * P::BN + g * P::LC * 4]);
#pragma unroll
            for (int i = 0; i < TR; ++i)
#pragma unroll
                for (int j = 0; j < TC; ++j) acc[i][j] += rM[i] * rN[j];
        }
    }
}

// k8 for block rows row0.. of C (the data-parallel part of k14).
template <class P>
__global__ void __launch_bounds__(P::NTH, P::MINB)
k_dp(int N, int ld, float *C, const float *A, const float *B, int row0)
{
    constexpr int TR = P::TR, TC = P::TC;
    extern __shared__ __align__(16) float smem[];
    float *As = smem, *Bs = smem + P::S * P::A_STAGE;
    const int rowOff = (blockIdx.y + row0) * P::BM, colOff = blockIdx.x * P::BN;
    float acc[TR][TC] = {};
    mainloop<P>(N, ld, A, B, As, Bs, rowOff, colOff, 0, (N + P::BK - 1) / P::BK, threadIdx.x, acc);
    store_big<P, false>(N, ld, C, rowOff, colOff, threadIdx.x, acc);
}

// Stream-K-style hybrid for the partial last wave. With T tiles and one block
// per SM on S SMs, whole waves leave the last one T % S tiles short. Here the
// first block rows of tiles (enough to hold the T % S leftover) are cut
// into PARTS equal K ranges, so that tiles * PARTS is a multiple of S, and S
// blocks work through the pieces part-major: at any step, all blocks are on
// the same K range of neighbouring tiles, so they share A and B in L2 like
// k8's blocks do. (A first version gave each block one contiguous run of K
// slices, true stream-K; the blocks then drifted to different K offsets, lost
// that sharing, and ran ~40% slower per tile: DRAM can't feed 48 blocks each
// streaming their own slices.) The block with a tile's last piece waits for
// the others (they ran at earlier steps) through a per-tile counter, adds
// their partial sums from a workspace, writes C and resets the counter. No
// memset of C. The remaining
// rows run afterwards as plain k8 tiles (k_dp).
struct StreamK {
    int sk_tiles, parts;
    float *ws;      // one partial tile per piece
    int *count;     // per stream-K tile, zero between launches
};

template <class P>
__global__ void __launch_bounds__(P::NTH, P::MINB)
k_streamk(int N, int ld, float *C, const float *A, const float *B, StreamK sk)
{
    constexpr int TR = P::TR, TC = P::TC, E = TR * TC;
    extern __shared__ __align__(16) float smem[];
    float *As = smem, *Bs = smem + P::S * P::A_STAGE;
    const int t = threadIdx.x, gx = (N + P::BN - 1) / P::BN, KT = (N + P::BK - 1) / P::BK;
    const int units = sk.sk_tiles * sk.parts;
    for (int u = blockIdx.x; u < units; u += gridDim.x) {
        const int part = u / sk.sk_tiles, tile = u % sk.sk_tiles;
        const int kb = KT * part / sk.parts, ke = KT * (part + 1) / sk.parts;
        const int rowOff = tile / gx * P::BM, colOff = tile % gx * P::BN;
        float acc[TR][TC] = {};
        mainloop<P>(N, ld, A, B, As, Bs, rowOff, colOff, kb, ke, t, acc);
        if (sk.parts == 1) {
            store_big<P, false>(N, ld, C, rowOff, colOff, t, acc);
            continue;
        }
        if (part < sk.parts - 1) {   // park the partial sums, then count in
            float *mine = sk.ws + (size_t)u * E * P::NTH;
#pragma unroll
            for (int i = 0; i < TR; ++i)
#pragma unroll
                for (int j = 0; j < TC; ++j) __stcg(&mine[(i * TC + j) * P::NTH + t], acc[i][j]);
            __threadfence();
            __syncthreads();
            if (t == 0) atomicAdd(&sk.count[tile], 1);
            continue;
        }
        // The last part owns the tile. The other parts ran at earlier steps, so
        // the wait is short (and every block is resident: one per SM).
        if (t == 0) {
            while (atomicAdd(&sk.count[tile], 0) < sk.parts - 1) __nanosleep(100);
            sk.count[tile] = 0;
        }
        __syncthreads();
        __threadfence();
        for (int p = 0; p < sk.parts - 1; ++p) {
            const float *o = sk.ws + ((size_t)p * sk.sk_tiles + tile) * E * P::NTH;
#pragma unroll
            for (int i = 0; i < TR; ++i)
#pragma unroll
                for (int j = 0; j < TC; ++j) acc[i][j] += __ldcg(&o[(i * TC + j) * P::NTH + t]);
        }
        store_big<P, false>(N, ld, C, rowOff, colOff, t, acc);
    }
}

// ---------------------------------------------------------------- harness

typedef void (*Kernel)(int, int, float *, const float *, const float *);

struct Variant {
    const char *name;
    Kernel fn;      // nullptr: pick a tile shape per size (k11), see launch
    bool block2d;   // k0/k1 use the course's 16x16 thread block
    int dsmem = 0;  // dynamic shared memory, bytes
    int bm = BM, bn = BN, nt = NT;
    int split = 1;  // K split (k9)
    void (*custom)(int, int, float *, const float *, const float *) = nullptr;   // own launcher (k14)
};

template <class P> constexpr Variant big(const char *name, int split = 1)
{
    return {name, split > 1 ? k_big<P, true> : k_big<P>, false, P::SMEM, P::BM, P::BN, P::NTH, split};
}
template <class P> constexpr Variant kmajor(const char *name)
{
    return {name, k_kmajor<P>, false, KMajor<P>::SMEM, P::BM, P::BN, P::NTH};
}

using Big = Shape<128, 256, 16, 2, 256, 64, 64, 4, 1>;     // 128x256 tile, 16x8 per thread
using Small = Shape<128, 128, 16, 2, 128, 64, 64, 4, 2>;   // 128x128 tile, 16x8 per thread
template <class P, int MAXR = 0> constexpr Variant kmajor2(const char *name)
{
    return {name, k_kmajor2<P, MAXR>, false, KMajor<P>::SMEM, P::BM, P::BN, P::NTH};
}

static int num_sms = 1;

template <class P>
static void launch_streamk(int n, int ld, float *C, const float *A, const float *B)
{
    static float *ws = nullptr;
    static int *count = nullptr;
    static size_t ws_bytes = 0;
    const int gx = (n + P::BN - 1) / P::BN, gy = (n + P::BM - 1) / P::BM, tiles = gx * gy;
    // Model the busiest SM's work in tiles: stream-K rows cut into `parts`
    // K-pieces (plus ~10% per piece for the fixup) followed by whole-tile
    // waves for the rest. Few parts only: small pieces lose to the fixup.
    const int kt = (n + P::BK - 1) / P::BK;
    int sk_rows = 0, parts = 1;
    double best = (double)((tiles + num_sms - 1) / num_sms);
    for (int r = 1; r <= gy; ++r)
        for (int p = 2; p <= 4 && kt / p >= 8; ++p) {
            const long pieces = (long)r * gx * p;
            const double w = (double)((pieces + num_sms - 1) / num_sms) / p * 1.1 +
                             (double)(((gy - r) * gx + num_sms - 1) / num_sms);
            if (w < best - 1e-9) best = w, sk_rows = r, parts = p;
        }
    const size_t need = (size_t)sk_rows * gx * parts * P::BM * P::BN * 4;
    if (!count) {
        CK(cudaMalloc(&count, 1 << 20));
        CK(cudaMemset(count, 0, 1 << 20));
        CK(cudaFuncSetAttribute(k_streamk<P>, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM));
        CK(cudaFuncSetAttribute(k_dp<P>, cudaFuncAttributeMaxDynamicSharedMemorySize, P::SMEM));
    }
    if (need > ws_bytes) {
        if (ws) CK(cudaFree(ws));
        CK(cudaMalloc(&ws, need));
        ws_bytes = need;
    }
    if (sk_rows) {
        StreamK sk{sk_rows * gx, parts, ws, count};
        k_streamk<P><<<num_sms, P::NTH, P::SMEM>>>(n, ld, C, A, B, sk);
    }
    if (sk_rows < gy)
        k_dp<P><<<dim3(gx, gy - sk_rows), P::NTH, P::SMEM>>>(n, ld, C, A, B, sk_rows);
}

static const Variant k8v = big<Big>("k8 +128x256, 16x8/thread");
static const Variant k9v = big<Big>("k9 k8 + split-K=3", 3);
static const Variant k10v = kmajor<Small>("k10 +k-major A, 128x128");

static void launch_k15(int n, int ld, float *C, const float *A, const float *B);

static const Variant variants[] = {
    {"k0 baseline", k0_baseline, true},
    {"k1 +launch_bounds", k1_launch, true},
    {"k2 +float4", k_single<32, Contig>, false},
    {"k3 +bank-conflict fix", k_single<32, Split>, false},
    {"k4 +warptile", k_single<32, Warp>, false},
    {"k4 +warptile (BK=16)", k_single<16, Warp>, false},
    {"k5 +double buffer (BK=16)", k_dbuf<16, Warp>, false},
    {"k6 +B pair swap", k_swap<16, Warp>, false},
    {"k7 +cp.async (2 st, BK=16)", k_async<16, 2, Warp>, false, 2 * AsyncTiles<16>::STAGE_BYTES},
    {"k7 +cp.async (4 st, BK=8)", k_async<8, 4, Warp>, false, 4 * AsyncTiles<8>::STAGE_BYTES},
    k8v,
    k9v,
    k10v,
    {"k11 pick k8/k9/k10 by size", nullptr, false},
    kmajor2<Big>("k12 k-major A, 128x256"),
    kmajor2<Small>("k12 k-major A, 128x128"),
    big<Quad<Big>>("k13 k8 + 4x4 half-warps"),
    big<Quad<Small>>("k13 128x128 + 4x4 half-warps"),
    {"k14 k8 + split last wave", k_big<Big>, false, Big::SMEM, Big::BM, Big::BN, Big::NTH, 1, launch_streamk<Big>},
    {"k15 k11 below a wave, else k14", k_big<Big>, false, Big::SMEM, Big::BM, Big::BN, Big::NTH, 1, launch_k15},
};

// k11: pick the variant whose busiest SM gets the least work. An SM runs its
// share of the blocks one after another (or side by side, at the same total
// rate), so its work is ceil(blocks / SMs) * BM * BN / split. Splitting K
// costs a memset and atomics, which measured slower whenever the unsplit grid
// already covers every SM, so it's only considered when it doesn't. Ties go
// to the earlier candidate.
static Variant pick(int n)
{
    auto work = [n](const Variant &v, int split) {
        long tiles = (long)((n + v.bm - 1) / v.bm) * ((n + v.bn - 1) / v.bn);
        long waves = (tiles * split + num_sms - 1) / num_sms;
        return (double)(waves * v.bm * v.bn) / split;
    };
    Variant best = k8v;
    double w = work(k8v, 1);
    if (work(k10v, 1) < w) best = k10v, w = work(k10v, 1);
    const long tiles = (long)((n + Big::BM - 1) / Big::BM) * ((n + Big::BN - 1) / Big::BN);
    for (int s = 2; s <= 4 && tiles < num_sms; ++s)
        if (work(k9v, s) < w) best = k9v, best.split = s, w = work(k9v, s);
    return best;
}

static void launch(const Variant &v0, int n, int ld, float *C, const float *A, const float *B)
{
    if (v0.custom) return v0.custom(n, ld, C, A, B);
    const Variant v = v0.fn ? v0 : pick(n);
    dim3 grid((n + v.bn - 1) / v.bn, (n + v.bm - 1) / v.bm, v.split);
    dim3 block = v.block2d ? dim3(16, 16) : dim3(v.nt);
    if (v.split > 1)
        CK(cudaMemsetAsync(C, 0, (size_t)n * ld * 4));
    v.fn<<<grid, block, v.dsmem>>>(n, ld, C, A, B);
}

// k15: fewer 128x256 tiles than SMs means k11's split-K or smaller tiles;
// otherwise k14 (whole waves plus a split last wave).
static void launch_k15(int n, int ld, float *C, const float *A, const float *B)
{
    const long tiles = (long)((n + Big::BM - 1) / Big::BM) * ((n + Big::BN - 1) / Big::BN);
    if (tiles < num_sms) return launch(Variant{"k11", nullptr, false}, n, ld, C, A, B);
    launch_streamk<Big>(n, ld, C, A, B);
}

// SGEMM_ONLY=k5,k6: run only kernels whose name starts with one of these.
static bool selected(const Variant &v)
{
    const char *only = getenv("SGEMM_ONLY");
    if (!only) return true;
    for (const char *p = only; *p;) {
        const char *e = strchr(p, ',');
        size_t n = e ? (size_t)(e - p) : strlen(p);
        if (n && strncmp(v.name, p, n) == 0) return true;
        p += n + (e != nullptr);
    }
    return false;
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
    num_sms = prop.multiProcessorCount;
    printf("# %s, %d SMs, CC %d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    for (const Variant &v : variants) {
        if (v.fn && v.dsmem > 48 * 1024)
            CK(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, v.dsmem));
        if (!selected(v)) continue;
        if (!v.fn) {
            printf("# %-28s picks k8, k9 or k10 per size\n", v.name);
            continue;
        }
        cudaFuncAttributes a;
        CK(cudaFuncGetAttributes(&a, v.fn));
        int blocks;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, v.fn, v.nt, v.dsmem));
        printf("# %-28s regs=%3d smem=%5zu B  blocks/SM=%d  spill_ld=%zu B\n", v.name,
               a.numRegs, a.sharedSizeBytes + v.dsmem, blocks, a.localSizeBytes);
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
            if (!selected(v)) continue;
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
