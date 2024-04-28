#include "ReLULayer.cuh"
#include "cudaHelpers.cuh"
#include "math.h"

__global__ void reluForwardKernel(reluInput_t inputs);
__global__ void reluBackwardKernel(float *upstreamGradients, reluInput_t inputs,
                                   float *gradientsOut);
                                   
reluInit_t reluInit(float* inputs, float* outputs, unsigned int dim) {
    reluInit_t *newReLU = (reluInit_t*)malloc(sizeof(reluInit_t*));
    newReLU->inputs = inputs;
    newReLU->outputs = outputs;
    newReLU->dim = dim;

    float* dev_inputs;
    gpuErrchk(cudaMalloc((float**)&dev_inputs, sizeof(reluInit_t));
}

void reluForward(reluInput_t *inputs) {
    // Set params for our kernel
    int blockDim = 128;
    int gridDim = ceil(1.0 * inputs->dim / blockDim);

    // Run cuda kernel for forward relu operation
    reluForwardKernel<<<gridDim, blockDim>>>(*inputs);
    gpuErrchk(cudaDeviceSynchronize());
}

void reluBackward(float *upstreamGradients, reluInput_t *inputs, float *gradientsOut) {
    // Set params for our kernel
    int blockDim = 128;
    int gridDim = ceil(1.0 * inputs->dim / blockDim);

    // Run cuda kernel for backward relu operation
    reluBackwardKernel<<<gridDim, blockDim>>>(upstreamGradients, *inputs, gradientsOut);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void reluForwardKernel(reluInput_t inputs) {
    // Assign tid
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Make sure thread is not out of bounds for our dim
    if (tid < inputs.dim) {
        // Set thread's output to 0 automatically
        inputs.outputs[tid] = 0;

        // Reset output to whatever is bigger: 0 or our input
        atomicMax(&(inputs.outputs[tid]), inputs.inputs[tid]);
    }

    // Operation done, end kernel
}

__global__ void reluBackwardKernel(float *upstreamGradients, reluInput_t inputs,
                                   float *gradientsOut) {
    // Assign tid
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Make sure thread isn't out of bounds for our dim
    if (tid < inputs.dim) {
        // Set gradientsOut to 0 automatically
        gradientsOut[tid] = 0;

        // Check if our input is greater than 0
        if (inputs.inputs[tid] > 0) {
            // Reset gradientsOut for our index to whatever our upstreamGradients is
            // (1 * upstreamGradients) = upstreamGradients
            atomicAdd(&gradientsOut[tid], upstreamGradients[tid]);
        }
    }

    // Operation done, end kernel
}
