#include "ReLULayer.cuh"
#include "cudaHelpers.cuh"
#include "math.h"

__global__ void reluForward(reluInput_t *inputs);
__global__ void reluBackward(float *upstreamGradients, reluInput_t *inputs, float *gradientsOut);

void reluForward(reluInput_t *inputs)
{
	// Set params for our kernel
    int blockDim = 128;
    int gridDim = ceil(1.0 * inputs->Dim / blockDim);

    // Run cuda kernel for forward relu operation
    reluForward<<<gridDim, blockDim>>>(inputs);
    gpuErrchk(cudaDeviceSynchronize());
}

void reluBackward(float *upstreamGradients, reluInput_t *inputs, float *gradientsOut)
{
	// Set params for our kernel
    int blockDim = 128;
    int gridDim = ceil(1.0 * inputs->Dim / blockDim);

	// Run cuda kernel for backward relu operation
    reluBackward<<<gridDim, blockDim>>>(upstreamGradients, inputs, gradientsOut);
    gpuErrchk(cudaDeviceSynchronize());
}

__global__ void reluForward(reluInput_t *inputs) 
{
	// Assign tid
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Make sure thread is not out of bounds for our dim
	if( tid < inputs->dim) 
	{
		// Set thread's output to 0 automatically
		inputs->outputs[index] = 0;

		// Reset output to whatever is bigger: 0 or our input
		AtomicMax(inputs->outputs[index], inputs->inputs[index]);
	}

	// Operation done, end kernel
}

__global__ void reluBackward(float *upstreamGradients, reluInput_t *inputs, float *gradientsOut) 
{
	// Assign tid
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Make sure thread isn't out of bounds for our dim
	if( tid < inputs->dim )
	{
		// Set gradientsOut to 0 automatically
	   *gradientsOut[index] = 0;

       // Check if our input is greater than 0
	   if( inputs->input[index] > 0)
	   {
		   // Reset gradientsOut for our index to whatever our upstreamGradients is
		   // (1 * upstreamGradients) = upstreamGradients
		   atomicAdd(&gradientsOut[index], *upstreamGradients);
	   }
	}

	// Operation done, end kernel
}

