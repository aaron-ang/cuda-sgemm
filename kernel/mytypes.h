#ifndef MYTYPES_H
#define MYTYPES_H

// tilescale (# of points computed by each thread)
#ifndef TILESCALE_M
#define TILESCALE_M 1 // Enter your own values
#endif
#ifndef TILESCALE_N
#define TILESCALE_N 1 // Enter your own values
#endif
#ifndef TILESCALE_K
#define TILESCALE_K 1 // Enter your own values
#endif

// Nvidia T4 Turing Architecture
// 255 registers per thread
// 64KB shared memory per SM
// 64k 32-bit registers per SM
// Max concurrent warps per SM: 32
// Max concurrent threads per SM: 32 * 32 = 1024
// Max thread blocks per SM: 16
// Max threads per block: 1024
// BLOCKDIM_X * BLOCKDIM_Y <= 1024 and multiple of 32
// Shared memory is divided into 32 banks, 4B per bank
// Max global memory transaction size: 128B -> 32 floats
#define BLOCKDIM BLOCKDIM_X
#define TILEDIM_M (BLOCKDIM * TILESCALE_M) // Enter your own values
#define TILEDIM_N (BLOCKDIM * TILESCALE_N) // Enter your own values

// matrix A loads with warps along the horiziontal axis (K)
// so to get good coalescaed loads, we want TILEDIM_K to be >= 32
#define TILEDIM_K 32 // Enter your own values

// step size in each dimension
#define TILESTEP_N 1 // Enter your own values
#define TILESTEP_K 1 // Enter your own values
#define TILESTEP_M 1 // Enter your own values

#endif
