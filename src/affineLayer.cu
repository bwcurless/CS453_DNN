#include "affineLayer.cuh"
#include "cudaHelpers.cuh"

// Prototypes for kernel and other private functions. Feel free to change them and their signatures
__global__ void affineForwardKernel(affineInputs_t inputs);
__global__ void affineBackwardKernel(float* dLdf, affineInputs_t aff1Inputs);
__global__ void affineUpdateKernel(learnParams_t hyperParams, affineInputs_t inputs);

// Host code will call these functions, and they will launch the kernels
// Moved this out of main code to make it easier to test, and not have to copy all the kernel launch
// parameters around
affineInputs_t* affineInit(unsigned int numOutputs, unsigned int batchSize,
                           unsigned int inputDataSize, float* x) {
    // W1. The weight matrix we are trying to find
    float* dev_W;
    gpuErrchk(cudaMalloc((float**)&dev_W, sizeof(float) * numOutputs * inputDataSize));

    // b1. The biases for each output of the linear classifier. The +b term
    float* dev_b;
    gpuErrchk(cudaMalloc((float**)&dev_b, sizeof(float) * numOutputs));

    // Intermediate Scores f(x). The linear classifier's predicted scores f(x)=W*x+b
    float* dev_f;
    gpuErrchk(cudaMalloc((float**)&dev_f, sizeof(float) * numOutputs));

    // dL/dW1. How much the weights effect the loss
    float* dev_dLdW;
    gpuErrchk(cudaMalloc((float**)&dev_dLdW, sizeof(float) * numOutputs * inputDataSize));

    // dL/db1. How much the biases effect the loss
    float* dev_dLdb;
    gpuErrchk(cudaMalloc((float**)&dev_dLdb, sizeof(float) * numOutputs));

    // dL/dx1. How much the inputs effect the loss
    float* dev_dLdx;
    gpuErrchk(cudaMalloc((float**)&dev_dLdx, sizeof(float) * batchSize * inputDataSize));

    affineInputs_t* aff1Inputs = (affineInputs_t*)malloc(sizeof(affineInputs_t));
    aff1Inputs->W = dev_W;
    aff1Inputs->x = x;
    aff1Inputs->b = dev_b;
    aff1Inputs->f = dev_f;
    aff1Inputs->dLdB = dev_dLdb;
    aff1Inputs->dLdW = dev_dLdW;
    aff1Inputs->dLdx = dev_dLdx;
    aff1Inputs->batchSize = batchSize;
    aff1Inputs->dataSize = inputDataSize;
    aff1Inputs->numOutputs = numOutputs;

    return aff1Inputs;
}

void affineForward(const affineInputs_t* inputs) {
    dim3 blockDim(32, 32);
    // Number of threads is the size of the output matrix
    dim3 gridDim(ceil(1.0 * inputs->dataSize / blockDim.x),
                 ceil(1.0 * inputs->numOutputs / blockDim.y));
    affineForwardKernel<<<gridDim, blockDim>>>(*inputs);
}

/*
void affineBackward(const float* upstreamGradient, const affineInputs_t* inputs,
                    const affineGradients_t* gradients) {
    dim3 blockDim(32, 32);
    // Number of threads is the size of the output matrix
    dim3 gridDim(ceil(1.0 * inputs->batchSize / blockDim.x),
                 ceil(1.0 * inputs->numOutputs / blockDim.y));
    affineBackwardKernel<<<gridDim, blockDim>>>(dev_dLdf, *aff1Inputs, *aff1Grads);
}

void affineUpdate(const learnParams_t* hyperParams, const affineInputs_t* inputs,
                  const affineGradients_t* gradients) {
    dim3 blockDim(32, 32);
    dim3 gridDim(ceil(1.0 * inputs->batchSize / blockDim.x),
                 ceil(1.0 * inputs->numOutputs / blockDim.y));
    affineUpdateKernel<<<gridDim, blockDim>>>(learnParameters, *aff1Inputs, *aff1Grads);
}
*/
// This could be good code to use as a starting point for creating the rectangular matrix
// multiply It could become the affineForward kernel

// matrix multiply
// each thread computes a single element of C using a row of A and column of B
// uses shared memory to tile the computation to eliminate extra accesses to
// global memory
// This is my own test for fun taking the original tiled MM and making it's accesses coalesced
__global__ void affineForwardKernel(float* A, float* B, float* C, const unsigned int NUMELEM) {
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
            // The first access is broadcase since all threads when blockdimtile is 32 have same
            // y, and k THe second access is spread out across banks since constant k, and
            // increasing x
            localSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // This one is coalesced as well
    C[ROW * NUMELEM + COL] = localSum;

    return;
}
