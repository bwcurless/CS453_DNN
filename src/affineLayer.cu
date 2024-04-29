#include "affineLayer.cuh"
#include "cudaHelpers.cuh"

// Prototypes for kernel and other private functions. Feel free to change them and their signatures
__global__ void affineForwardKernel(affineInputs_t inputs);

__global__ void dxKernel(const float* dLdf, affineInputs_t inputs);
__global__ void dWdbKernel(const float regularizationStrength, const float* dLdf,
                           affineInputs_t inputs);

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
    gpuErrchk(cudaMalloc((float**)&dev_f, sizeof(float) * numOutputs * batchSize));

    // dL/dW1. How much the weights effect the loss
    float* dev_dLdW;
    gpuErrchk(cudaMalloc((float**)&dev_dLdW, sizeof(float) * numOutputs * inputDataSize));

    // Momentum (old averaged gradients), for W and b
    float* dev_m;
    gpuErrchk(
        cudaMalloc((float**)&dev_m, sizeof(float) * (numOutputs * inputDataSize + (numOutputs))));

    // dL/db1. How much the biases effect the loss
    float* dev_dLdb;
    gpuErrchk(cudaMalloc((float**)&dev_dLdb, sizeof(float) * numOutputs));

    // dL/dx1. How much the inputs effect the loss
    float* dev_dLdx;
    gpuErrchk(cudaMalloc((float**)&dev_dLdx, sizeof(float) * batchSize * inputDataSize));

    // Regularization Loss. Make sure W doesn't get too big
    float* dev_regLoss;
    gpuErrchk(cudaMalloc((float**)&dev_regLoss, sizeof(float)));

    affineInputs_t* aff1Inputs = (affineInputs_t*)malloc(sizeof(affineInputs_t));
    aff1Inputs->W = dev_W;
    aff1Inputs->x = x;
    aff1Inputs->b = dev_b;
    aff1Inputs->f = dev_f;
    aff1Inputs->dLdB = dev_dLdb;
    aff1Inputs->dLdW = dev_dLdW;
    aff1Inputs->dLdx = dev_dLdx;
    aff1Inputs->regLoss = dev_regLoss;
    aff1Inputs->m = dev_m;
    aff1Inputs->batchSize = batchSize;
    aff1Inputs->dataSize = inputDataSize;
    aff1Inputs->numOutputs = numOutputs;

    return aff1Inputs;
}

void affineForward(const affineInputs_t* inputs) {
    dim3 blockDim(32, 32);
    // Number of threads is the size of the output matrix
    dim3 gridDim(ceil(1.0 * inputs->batchSize / blockDim.x),
                 ceil(1.0 * inputs->numOutputs / blockDim.y));
    affineForwardKernel<<<gridDim, blockDim>>>(*inputs);
    gpuErrchk(cudaDeviceSynchronize());
}

void affineBackward(const float regularizationStrength, const float* upstreamGradient,
                    const affineInputs_t* inputs) {
    // dL/dx and dL/dW are really just matrix multiplication kernels, so launch them with square
    // grids
    // Compute dL/dx
    dim3 blockDim(32, 32);
    // Number of threads is the size of the output matrix
    // Output is of size matching x
    dim3 gridDim(ceil(1.0 * inputs->batchSize / blockDim.x),
                 ceil(1.0 * inputs->dataSize / blockDim.y));
    dxKernel<<<gridDim, blockDim>>>(upstreamGradient, *inputs);

    // Compute dL/dW
    // Compute dL/db
    dim3 blockDim2(32, 32);
    // Number of threads is the size of the output matrix
    // Output is of size matching W
    dim3 gridDim2(ceil(1.0 * inputs->dataSize / blockDim.x),
                  ceil(1.0 * inputs->numOutputs / blockDim.y));
    dWdbKernel<<<gridDim2, blockDim2>>>(regularizationStrength, upstreamGradient, *inputs);
    gpuErrchk(cudaDeviceSynchronize());
}

void affineUpdate(const learnParams_t* hyperParams, const affineInputs_t* inputs) {
    dim3 blockDim(128, 1);
    dim3 gridDim(ceil(1.0 * inputs->dataSize * inputs->numOutputs / blockDim.x), 1);
    affineUpdateKernel<<<gridDim, blockDim>>>(*hyperParams, *inputs);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void affineForwardKernel(affineInputs_t inputs) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    float localSum = 0;

    if (col < inputs.batchSize && row < inputs.numOutputs) {
        for (unsigned int index = 0; index < inputs.dataSize; index++) {
            localSum +=
                inputs.W[row * inputs.dataSize + index] * inputs.x[index * inputs.batchSize + col];
        }
        inputs.f[row * inputs.batchSize + col] = localSum + inputs.b[row];
    }

    return;
}

// dL/dW is a matrix multiply dL/df * x^T
// dL/db is a row wise sum of upstream gradients, we already read dLdf across the row, so compute
// the sum here as well
// Computes the effect of regularization on W as well
__global__ void dWdbKernel(const float regularizationStrength, const float* dLdf,
                           affineInputs_t inputs) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    float localSum = 0;
    float localdBSum = 0;

    // Output is (numOutputs, inputSize), same size as W
    if (row < inputs.numOutputs && col < inputs.dataSize) {
        for (int i = 0; i < inputs.batchSize; i++) {
            // Access dLdf like normal, access x as x^T, so read across col'th row
            float grad = dLdf[row * inputs.batchSize + i];
            float localGradient = inputs.x[col * inputs.batchSize + i];
            localSum += grad * localGradient;
            localdBSum += grad;
        }
        // Factor in regularization penalty here - dL/dW of L2 Norm of W
        // Ends up being 2 * regStrength * W
        localSum += 2 * regularizationStrength * inputs.W[row * inputs.dataSize + col];

        inputs.dLdW[row * inputs.dataSize + col] = localSum;
        // All threads compute the sum for dB, since we've already read the value we need, but we
        // really only need the first column to store it out. Not sure if this matters for
        // performance at all, but don't think it will hurt
        if (col == 0) {
            inputs.dLdB[row] = localdBSum;
        }
        // Lazy way to compute regularization loss
        float w = inputs.W[row * inputs.dataSize + col];

        atomicAdd(inputs.regLoss, regularizationStrength * w * w);
    }
}

// dL/dx is a matrix multiply W^T * dL/df
__global__ void dxKernel(const float* dLdf, affineInputs_t inputs) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    float localSum = 0;

    // Kind of a speed hack to clear this here since we need it in the next kernel
    if (col == 0 and row == 0) {
        *inputs.regLoss = 0.0;
    }

    // Output is (inputSize, batchSize), same size as x
    if (row < inputs.dataSize && col < inputs.batchSize) {
        for (int i = 0; i < inputs.numOutputs; i++) {
            // Access W as W^T, so read across row'th col, Access dLdf like normal,
            localSum += inputs.W[i * inputs.dataSize + row] * dLdf[i * inputs.batchSize + col];
        }
        inputs.dLdx[row * inputs.batchSize + col] = localSum;
    }
}

// Take a step by learningrate * - gradient
// Uses momentum based learning where we keep track of a decayed sum of previous gradients, and take
// a step in that direction
__global__ void affineUpdateKernel(learnParams_t hyperParams, affineInputs_t inputs) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < inputs.dataSize * inputs.numOutputs) {
        // Update W
        float step = -hyperParams.learningRate * inputs.dLdW[tid];
        float newGrad = inputs.m[tid] * hyperParams.momentumDecay + step;
        // Save it for next iteration
        inputs.m[tid] = newGrad;
        inputs.W[tid] += newGrad;
        if (tid < inputs.numOutputs) {
            // Update B
            float step = -hyperParams.learningRate * inputs.dLdB[tid];
            int bIndex = inputs.numOutputs * inputs.batchSize + tid;
            float newGrad = inputs.m[bIndex] * hyperParams.momentumDecay + step;
            // Save it for next iteration
            inputs.m[bIndex] = newGrad;
            inputs.b[tid] += newGrad;
        }
    }
}

// This could be good code to use as a starting point for creating the rectangular matrix
// multiply It could become the affineForward kernel

// matrix multiply
// each thread computes a single element of C using a row of A and column of B
// uses shared memory to tile the computation to eliminate extra accesses to
// global memory
// This is my own test for fun taking the original tiled MM and making it's accesses coalesced
//__global__ void affineForwardKernel(float* A, float* B, float* C, const unsigned int NUMELEM)
//{
//    // Copy code from in-class activity
//
//    unsigned int COL = threadIdx.x + blockDim.x * blockIdx.x;
//    unsigned int ROW = threadIdx.y + blockDim.y * blockIdx.y;
//
//    __shared__ float tileA[BLOCKDIMTILE][BLOCKDIMTILE];
//    __shared__ float tileB[BLOCKDIMTILE][BLOCKDIMTILE];
//
//    float localSum = 0;
//
//    for (int phase = 0; phase < NUMELEM; phase += BLOCKDIMTILE) {
//        // Both accesses are coalesced here
//        tileA[threadIdx.y][threadIdx.x] = A[ROW * NUMELEM + phase + threadIdx.x];
//        tileB[threadIdx.y][threadIdx.x] = B[(phase + threadIdx.y) * NUMELEM + COL];
//
//        __syncthreads();
//
//        for (int k = 0; k < BLOCKDIMTILE; k++) {
//            // The first access is broadcase since all threads when blockdimtile is 32 have
//            same
//            // y, and k THe second access is spread out across banks since constant k, and
//            // increasing x
//            localSum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
//        }
//
//        __syncthreads();
//    }
//
//    // This one is coalesced as well
//    C[ROW * NUMELEM + COL] = localSum;
//
//    return;
//}
