#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <iostream>
#include <random>

#include "../../src/affineLayer.cuh"
#include "../../src/cudaHelpers.cuh"

#define INPUTSIZE 10
#define NUMOUTPUTS 15
#define MINIBATCHSIZE 5

using namespace std;

// function prototypes
void warmUpGPU();
void printMatrix(float *matrix, int width, int height);
void outputSumElems(float *C, unsigned int NUMELEM);

int main(int argc, char *argv[]) {
    warmUpGPU();

    // seed random number generator with constant seed
    srand(123);

    // Init weight matrix
    int WSize = sizeof(float) * NUMOUTPUTS * INPUTSIZE;
    float *host_W = (float *)malloc(WSize);
    // Init to increasing number for ease of computation
    for (int i = 0; i < NUMOUTPUTS * INPUTSIZE; i++) {
        host_W[i] = i;
    }

    // Init input values
    size_t inputDataSize = sizeof(float) * INPUTSIZE * MINIBATCHSIZE;
    float *host_inputData = (float *)malloc(inputDataSize);
    for (int i = 0; i < INPUTSIZE * MINIBATCHSIZE; i++) {
        host_inputData[i] = 0.25;
    }

    // Make space for input data on GPU
    // This comes from previous layer, so it won't be allocated with the rest of the other layer
    // data
    float *dev_inputData;
    gpuErrchk(cudaMalloc((float **)&dev_inputData, WSize));

    // Allocate GPU space for rest of affine layer
    affineInputs_t *affineInputs = affineInit(NUMOUTPUTS, MINIBATCHSIZE, INPUTSIZE, dev_inputData);

    // Push initial data to GPU, x and W
    gpuErrchk(cudaMemcpy(affineInputs->W, host_W, WSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(affineInputs->x, host_inputData, inputDataSize, cudaMemcpyHostToDevice));

    double tstart = omp_get_wtime();

    // execute kernel
    affineForward(affineInputs);

    double tend = omp_get_wtime();

    // Copy outputs off GPU
    // I'll leave this up to you Evan

    // float *host_gradient;
    // host_gradient = (float *)malloc(sizeof(float) * NUMOUTPUTS * MINIBATCHSIZE);
    // gpuErrchk(cudaMemcpy(host_gradient, softmaxInputs->dLdf,
    //                     sizeof(float) * NUMOUTPUTS * MINIBATCHSIZE, cudaMemcpyDeviceToHost));
    // printf("dL/df: \n");
    // printMatrix(host_gradient, MINIBATCHSIZE, NUMOUTPUTS);

    printf("\nTotal time GPU (s): %f", tend - tstart);

    printf("\n");

    // free memory

    return 0;
}

void warmUpGPU() {
    printf("Warming up GPU for time trialing...\n");
    cudaDeviceSynchronize();
    return;
}

void printMatrix(float *matrix, int width, int height) {
    int i, j;
    int cnt = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            printf("%.2f, ", matrix[cnt]);
            cnt++;
        }
        printf("\n");
    }
}
