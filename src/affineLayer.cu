
// This could be good code to use as a starting point for creating the rectangular matrix multiply

// My MODE 7 for fun
// matrix multiply
// each thread computes a single element of C using a row of A and column of B
// uses shared memory to tile the computation to eliminate extra accesses to
// global memory
// This is my own test for fun taking the original tiled MM and making it's accesses coalesced
__global__ void matrixMultiOneElemPerThreadSharedMemoryTileCoalesced(float *A, float *B, float *C,
                                                                     const unsigned int NUMELEM) {
    // Copy code from in-class activity

    unsigned int COL = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ROW = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
    __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];

    float localSum = 0;

    for (int phase = 0; phase < NUMELEM; phase += BLOCKDIMTILE) {
        // Both accesses are coalesced here
        tileA[threadIdx.y][threadIdx.x] = A[ROW * NUMELEM + phase + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(phase + threadIdx.y) * NUMELEM + COL];

        __syncthreads();

        for (int k = 0; k < BLOCKDIMTILE; k++) {
            // The first access is broadcase since all threads when blockdimtile is 32 have same y,
            // and k THe second access is spread out across banks since constant k, and increasing x
            localSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // This one is coalesced as well
    C[ROW * NUMELEM + COL] = localSum;

    return;
}
