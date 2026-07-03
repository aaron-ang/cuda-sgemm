// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;

    if ((I < N) && (J < N))
    {
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++)
        {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
// You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *__restrict__ C, _FTYPE_ *__restrict__ A, _FTYPE_ *__restrict__ B)
{
    // how many values in a blocktile (calculated by a thread block)
    const uint totalResultsBlocktile = TILEDIM_M * TILEDIM_N;
    // no. elements each thread calculates in the blocktile
    const uint numResultsPerThread = TILESCALE_M * TILESCALE_N; // 8*8
    // threads needed to calculate the blocktile
    const uint numThreadsBlocktile = totalResultsBlocktile / (numResultsPerThread);
    // numThreadsBlocktile should equal to blockDim.x * blockDim.y
    assert(numThreadsBlocktile == blockDim.x * blockDim.y);

    const uint gridRow = blockIdx.y;
    const uint gridCol = blockIdx.x;
    const uint globalRowOffset = gridRow * TILEDIM_M;
    const uint globalColOffset = gridCol * TILEDIM_N;

    // Move blocktile to beginning of A's row and B's column
    A += globalRowOffset * N;                   // make sense
                                                // TILEDIM_M
                                                // TILEDIM_M
                                                // TILEDIM_M
                                                //  A
    B += globalColOffset;                       // make sense TILEDIM_M TILEDIM_M B
    C += globalRowOffset * N + globalColOffset; // make sense       TILEDIM_N TILEDIM_N
                                                //        TILEDIM_M                     |
                                                //        TILEDIM_M                     |
                                                //        TILEDIM_M                     |
                                                //                  --------------------C

    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;

    // In one iteration, a maximum of numThreadsBlocktile values are loaded into shared mem.
    // Find out the inner row and column for each thread to load per iteration.
    const uint linearThreadIdx = threadRow * blockDim.x + threadCol;
    const uint innerRowA = linearThreadIdx / TILEDIM_K;
    const uint innerColA = linearThreadIdx % TILEDIM_K;
    const uint innerRowB = linearThreadIdx / TILEDIM_N;
    const uint innerColB = linearThreadIdx % TILEDIM_N;

    // Calculate the stride length required for all thread to load the tile.
    // Each thread loads values in a column of the sub blocktile.
    // In each iteration, numThreadsBlocktile / TILEDIM_K rows are loaded in shared mem A
    // and numThreadsBlocktile / TILEDIM_N rows are loaded in shared mem B.
    const uint strideA = numThreadsBlocktile / TILEDIM_K;
    const uint strideB = numThreadsBlocktile / TILEDIM_N;

    // register caches to calculate outer product
    // TILESCALE_M == TILESCALE_N
    _FTYPE_ threadResults[TILESCALE_M][TILESCALE_N] = {0.0f};
    _FTYPE_ regM[TILESCALE_M] = {0.0f};
    _FTYPE_ regN[TILESCALE_N] = {0.0f};

    // outer-most loop over sub blocktiles
    for (uint bkIdx = 0; bkIdx < N; bkIdx += TILEDIM_K)
    {
        // allocate space for the current sub blocktile in smem. Total size <= 64KB
        extern __shared__ _FTYPE_ sMem[];
        _FTYPE_(*As)
        [TILEDIM_K] = (_FTYPE_(*)[TILEDIM_K])sMem; // <= 32KB
        _FTYPE_(*Bs)
        [TILEDIM_N] = (_FTYPE_(*)[TILEDIM_N]) & sMem[TILEDIM_M * TILEDIM_K]; // <= 32KB

        // populate the SMEM caches
#pragma unroll
        for (uint loadOffset = 0; loadOffset < TILEDIM_M; loadOffset += strideA)
        {
            const uint globalRowA = globalRowOffset + innerRowA + loadOffset;
            const uint globalColA = bkIdx + innerColA;
            As[innerRowA + loadOffset][innerColA] = (globalRowA < N && globalColA < N)
                                                        ? A[(innerRowA + loadOffset) * N + innerColA]
                                                        : 0.0f;
        }
#pragma unroll
        for (uint loadOffset = 0; loadOffset < TILEDIM_K; loadOffset += strideB)
        {
            const uint globalRowB = bkIdx + innerRowB + loadOffset;
            const uint globalColB = globalColOffset + innerColB;
            Bs[innerRowB + loadOffset][innerColB] = (globalRowB < N && globalColB < N)
                                                        ? B[(innerRowB + loadOffset) * N + innerColB]
                                                        : 0.0f;
        }
        __syncthreads();

        // calculate per-thread
        // each thread computes a TILESCALE_M x TILESCALE_N contiguous submatrix
#pragma unroll
        for (uint dotIdx = 0; dotIdx < TILEDIM_K; ++dotIdx)
        {
            // load into registers
#pragma unroll
            for (uint i = 0; i < TILESCALE_M; ++i)
            {
                regM[i] = As[threadRow * TILESCALE_M + i][dotIdx];
            }
#pragma unroll
            for (uint i = 0; i < TILESCALE_N; ++i)
            {
                regN[i] = Bs[dotIdx][threadCol * TILESCALE_N + i];
            }
            // accumulate outer product
#pragma unroll
            for (uint resIdxM = 0; resIdxM < TILESCALE_M; ++resIdxM)
            {
#pragma unroll
                for (uint resIdxN = 0; resIdxN < TILESCALE_N; ++resIdxN)
                {
                    threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();

        // advance blocktile
        A += TILEDIM_K;     // move TILEDIM_K columns to right
        B += TILEDIM_K * N; // move TILEDIM_K rows down
    }

    // write out the results
#pragma unroll
    for (uint resIdxM = 0; resIdxM < TILESCALE_M; ++resIdxM)
    {
#pragma unroll
        for (uint resIdxN = 0; resIdxN < TILESCALE_N; ++resIdxN)
        {
            const uint tileRow = threadRow * TILESCALE_M + resIdxM;
            const uint tileCol = threadCol * TILESCALE_N + resIdxN;
            if (globalRowOffset + tileRow < N && globalColOffset + tileCol < N)
            {
                C[tileRow * N + tileCol] = threadResults[resIdxM][resIdxN];
            }
        }
    }
}

__global__ void matMulNonSquare(int M, int N, int K, _FTYPE_ *__restrict__ C, _FTYPE_ *__restrict__ A, _FTYPE_ *__restrict__ B)
{
    const uint totalResultsBlocktile = TILEDIM_M * TILEDIM_N;
    const uint numResultsPerThread = TILESCALE_M * TILESCALE_N;
    const uint numThreadsBlocktile = totalResultsBlocktile / (numResultsPerThread);
    assert(numThreadsBlocktile == blockDim.x * blockDim.y);

    const uint gridRow = blockIdx.y;
    const uint gridCol = blockIdx.x;
    const uint globalRowOffset = gridRow * TILEDIM_M;
    const uint globalColOffset = gridCol * TILEDIM_N;

    A += globalRowOffset * K;
    B += globalColOffset;
    C += globalRowOffset * N + globalColOffset;

    const int threadCol = threadIdx.x;
    const int threadRow = threadIdx.y;

    const uint linearThreadIdx = threadRow * blockDim.x + threadCol;
    const uint innerRowA = linearThreadIdx / TILEDIM_K;
    const uint innerColA = linearThreadIdx % TILEDIM_K;
    const uint innerRowB = linearThreadIdx / TILEDIM_N;
    const uint innerColB = linearThreadIdx % TILEDIM_N;

    const uint strideA = numThreadsBlocktile / TILEDIM_K;
    const uint strideB = numThreadsBlocktile / TILEDIM_N;

    _FTYPE_ threadResults[TILESCALE_M][TILESCALE_N] = {0.0f};
    _FTYPE_ regM[TILESCALE_M] = {0.0f};
    _FTYPE_ regN[TILESCALE_N] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += TILEDIM_K)
    {
        extern __shared__ _FTYPE_ sMem[];
        _FTYPE_(*As)
        [TILEDIM_K] = (_FTYPE_(*)[TILEDIM_K])sMem;
        _FTYPE_(*Bs)
        [TILEDIM_N] = (_FTYPE_(*)[TILEDIM_N]) & sMem[TILEDIM_M * TILEDIM_K];

#pragma unroll
        for (uint loadOffset = 0; loadOffset < TILEDIM_M; loadOffset += strideA)
        {
            const uint globalRowA = globalRowOffset + innerRowA + loadOffset;
            const uint globalColA = bkIdx + innerColA;
            As[innerRowA + loadOffset][innerColA] = (globalRowA < M && globalColA < K)
                                                        ? A[(innerRowA + loadOffset) * K + innerColA]
                                                        : 0.0f;
        }
#pragma unroll
        for (uint loadOffset = 0; loadOffset < TILEDIM_K; loadOffset += strideB)
        {
            const uint globalRowB = bkIdx + innerRowB + loadOffset;
            const uint globalColB = globalColOffset + innerColB;
            Bs[innerRowB + loadOffset][innerColB] = (globalRowB < K && globalColB < N)
                                                        ? B[(innerRowB + loadOffset) * N + innerColB]
                                                        : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (uint dotIdx = 0; dotIdx < TILEDIM_K; ++dotIdx)
        {
#pragma unroll
            for (uint i = 0; i < TILESCALE_M; ++i)
            {
                regM[i] = As[threadRow * TILESCALE_M + i][dotIdx];
            }
#pragma unroll
            for (uint i = 0; i < TILESCALE_N; ++i)
            {
                regN[i] = Bs[dotIdx][threadCol * TILESCALE_N + i];
            }
#pragma unroll
            for (uint resIdxM = 0; resIdxM < TILESCALE_M; ++resIdxM)
            {
#pragma unroll
                for (uint resIdxN = 0; resIdxN < TILESCALE_N; ++resIdxN)
                {
                    threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();

        A += TILEDIM_K;
        B += TILEDIM_K * N;
    }

#pragma unroll
    for (uint resIdxM = 0; resIdxM < TILESCALE_M; ++resIdxM)
    {
#pragma unroll
        for (uint resIdxN = 0; resIdxN < TILESCALE_N; ++resIdxN)
        {
            const uint tileRow = threadRow * TILESCALE_M + resIdxM;
            const uint tileCol = threadCol * TILESCALE_N + resIdxN;
            if (globalRowOffset + tileRow < M && globalColOffset + tileCol < N)
            {
                C[tileRow * N + tileCol] = threadResults[resIdxM][resIdxN];
            }
        }
    }
}
#endif
