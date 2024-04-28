#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <random>

#include "../../src/cudaHelpers.cuh"
#include "../../src/ReLULayer.cuh"

#define DIM 1000

void printMatrix(float *matrix, int width, int height);
void warmUpGPU();

int main(int argc, char *argv[]) {
	warmUpGPU();

	// Initialize variables
	float* inputArray, *outputArray;
	float* upstreamGradients, *gradientsOut;
	int index;
    bool testSuccess = true;

    // seed random number generator with constant seed
    srand(123);

	// Generate input & output array to test with
	for( index = 0; index < DIM; index++)
	{
		inputArray[index] = index + 1;
		outputArray[index] = 0;
	}

    // Generate upstreamGradients to test with
    for( index = 0; index < DIM; index++)
    {
        upstreamGradients[index] = index;
        gradientsOut[index] = 0;
    }

	// Push array to GPU
	reluInput_t inputs = reluInit(inputArray, outputArray, DIM);

	//*****RELU FORWARD TEST*****//

    // Start time to test GPU performance
    double tstart = omp_get_wtime();

    // execute kernel reluForward
    reluForward(&inputs);

    // End time for GPU performance
    double tend = omp_get_wtime();

	// Display Results
	printf("ReLUForward Operation completed in %f time\n", tend - tstart);
    for(index = 0; index < DIM; index++)
    {
        if(inputs.outputs[index] == 0)
        {
            printf("Output 0 found at index %d\n", index);
            testSuccess = false;
        }
    }
    if( testSuccess )
    {
        printf("Output behaved correctly, reluForward operational\n");
    }

	//*****RELU BACKWARD TEST*****//

    // reset flag
    testSuccess = true;

    // Start time to test GPU performance
    tstart = omp_get_wtime();

    // execute kernel reluForward
    reluBackward(upstreamGradients, &inputs, gradientsOut);

    // End time for GPU performance
    tend = omp_get_wtime();

	// Display Results
	printf("ReLUBackward operation completed in %f time\n", tend - tstart);
    for(index = 0; index < DIM; index++);
    {
        if(gradientsOut[index] == 0)
        {
            printf("0 found in gradientsOut at index %d\n", index);
            testSuccess = false;
        }
    }
    if( testSuccess )
    {
        printf("Output behaved correctly, reluBackward operational\n");
    }

    // free memory

    return 0;
}

void warmUpGPU() {
    printf("Warming up GPU for time trialing...\n");
    cudaDeviceSynchronize();
    return;
}
