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
#define NUMOUTPUTS 7
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
    printf("W:\n");
    printMatrix(host_W, INPUTSIZE, NUMOUTPUTS);

    // Init input values
    size_t inputDataSize = sizeof(float) * INPUTSIZE * MINIBATCHSIZE;
    float *host_inputData = (float *)malloc(inputDataSize);
    for (int i = 0; i < INPUTSIZE * MINIBATCHSIZE; i++) {
        host_inputData[i] = i;
    }
    printf("x:\n");
    printMatrix(host_inputData, MINIBATCHSIZE, INPUTSIZE);

    // Init upstream gradient
    size_t fSize = sizeof(float) * NUMOUTPUTS * MINIBATCHSIZE;
    float *host_dLdf = (float *)malloc(fSize);
    // Init to pseudo identity matrix to make checking easier
    for (int row = 0; row < NUMOUTPUTS; row++) {
        for (int col = 0; col < MINIBATCHSIZE; col++) {
            if (row == col) {
                host_dLdf[row * MINIBATCHSIZE + col] = 1;
            } else {
                host_dLdf[row * MINIBATCHSIZE + col] = 0;
            }
        }
    }
    printf("dLdf:\n");
    printMatrix(host_dLdf, MINIBATCHSIZE, NUMOUTPUTS);

    // Make space for input data on GPU
    // This comes from previous layer, so it won't be allocated with the rest of the other layer
    // data
    float *dev_inputData;
    gpuErrchk(cudaMalloc((float **)&dev_inputData, WSize));

    // Allocate GPU space for rest of affine layer
    affineInputs_t *affineInputs = affineInit(NUMOUTPUTS, MINIBATCHSIZE, INPUTSIZE, dev_inputData);

    // Make space for upstream gradients on GPU
    // This comes from previous layer, so it won't be allocated with the rest of the other layer
    // data
    float *dev_upstreamGradients;
    gpuErrchk(cudaMalloc((float **)&dev_upstreamGradients, fSize));

    // Push initial data to GPU, x and W
    gpuErrchk(cudaMemcpy(affineInputs->W, host_W, WSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(affineInputs->x, host_inputData, inputDataSize, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_upstreamGradients, host_dLdf, fSize, cudaMemcpyHostToDevice));

    double tstart = omp_get_wtime();

    // execute kernel
    affineForward(affineInputs);

    double tend = omp_get_wtime();

    double tstartBackward = omp_get_wtime();

    // execute kernel
    affineBackward(0.0, dev_upstreamGradients, affineInputs);

    double tendBackward = omp_get_wtime();

    // Copy outputs off GPU
    float *host_f;
    host_f = (float *)malloc(fSize);
    gpuErrchk(cudaMemcpy(host_f, affineInputs->f, fSize, cudaMemcpyDeviceToHost));
    printf("f:\n");
    printMatrix(host_f, MINIBATCHSIZE, NUMOUTPUTS);

    // Copy Gradients outputs off GPU
    float *host_dLdW;
    host_dLdW = (float *)malloc(WSize);
    gpuErrchk(cudaMemcpy(host_dLdW, affineInputs->dLdW, WSize, cudaMemcpyDeviceToHost));
    printf("dLdW:\n");
    printMatrix(host_dLdW, INPUTSIZE, NUMOUTPUTS);

    float *host_dLdx;
    size_t xSize = sizeof(float) * MINIBATCHSIZE * INPUTSIZE;
    host_dLdx = (float *)malloc(xSize);
    gpuErrchk(cudaMemcpy(host_dLdx, affineInputs->dLdx, xSize, cudaMemcpyDeviceToHost));
    printf("dLdx:\n");
    printMatrix(host_dLdx, MINIBATCHSIZE, INPUTSIZE);

    float *host_dLdb;
    size_t bSize = sizeof(float) * NUMOUTPUTS;
    host_dLdb = (float *)malloc(bSize);
    gpuErrchk(cudaMemcpy(host_dLdb, affineInputs->dLdB, bSize, cudaMemcpyDeviceToHost));
    printf("dLdb:\n");
    printMatrix(host_dLdb, 1, NUMOUTPUTS);

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
