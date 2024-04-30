#include "ReLULayer.cuh"
#include "cudaHelpers.cuh"
#include "math.h"

__global__ void reluForwardKernel(reluInput_t inputs);
__global__ void reluBackwardKernel(float *upstreamGradients, reluInput_t inputs,
                                   float *gradientsOut);

reluInput_t *reluInit(float *inputs, unsigned int dim) {
    float *dev_outputs;
    gpuErrchk(cudaMalloc((float **)&dev_outputs, sizeof(float) * dim));

    reluInput_t *newReLU = (reluInput_t *)malloc(sizeof(reluInput_t));
    newReLU->inputs = inputs;
    newReLU->outputs = dev_outputs;
    newReLU->dim = dim;

    return newReLU;
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
        // Reset output to whatever is bigger: 0 or our input
        if (inputs.inputs[tid] > 0) {
            inputs.outputs[tid] = inputs.inputs[tid];
        } else {
            inputs.outputs[tid] = 0;
        }
    }

    // Operation done, end kernel
}

__global__ void reluBackwardKernel(float *upstreamGradients, reluInput_t inputs,
                                   float *gradientsOut) {
    // Assign tid
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Make sure thread isn't out of bounds for our dim
    if (tid < inputs.dim) {
        // Check if our input was greater than 0
        if (inputs.inputs[tid] > 0) {
            // Reset gradientsOut for our index to whatever our upstreamGradients is
            // (1 * upstreamGradients) = upstreamGradients
            gradientsOut[tid] = upstreamGradients[tid];
        } else {
            // Kill the gradient if input was less than 0
            gradientsOut[tid] = 0;
        }
    }
}
