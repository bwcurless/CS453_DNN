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
#include "../../src/softmaxLoss.cuh"

#define MINIBATCHSIZE 100
#define CLASSES 10

using namespace std;

// function prototypes
void warmUpGPU();
void compareMatrices(float *C_GPU, float *C_CPU, unsigned int NUMELEM);
void printMatrix(float *matrix, int width, int height);
void outputSumElems(float *C, unsigned int NUMELEM);

int main(int argc, char *argv[]) {
    warmUpGPU();

    // seed random number generator with constant seed
    srand(123);

    // Intermediate Scores f(x). The linear classifier's predicted scores f(x)=W*x+b
    float *dev_f1;
    gpuErrchk(cudaMalloc((float **)&dev_f1, sizeof(float) * CLASSES));

    softmaxLoss_t *softmaxInputs = softmaxInit(CLASSES, MINIBATCHSIZE, dev_f1);

    // Set scores to small values, gaussian distribution, 0 mean
    float scale = 1.0;
    int scoresSize = sizeof(float) * CLASSES * MINIBATCHSIZE;
    float *host_scores = (float *)malloc(scoresSize);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, scale);

    for (int i = 0; i < CLASSES * MINIBATCHSIZE; i++) {
        host_scores[i] = distribution(generator);
    }
    // printMatrix(host_scores, MINIBATCHSIZE, CLASSES);

    // Copy scores to device
    gpuErrchk(cudaMemcpy(dev_f1, host_scores, scoresSize, cudaMemcpyHostToDevice));

    double tstart = omp_get_wtime();

    // execute kernel
    softmaxLoss(softmaxInputs);

    double tend = omp_get_wtime();

    // Copy Loss off GPU
    float *host_loss;
    host_loss = (float *)malloc(sizeof(float));
    gpuErrchk(cudaMemcpy(host_loss, dev_softmax_loss, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Softmax Loss: %f\n", *host_loss);

    // Copy gradient off GPU
    float *host_gradient;
    host_gradient = (float *)malloc(sizeof(float) * CLASSES);
    gpuErrchk(cudaMemcpy(host_gradient, dev_dLdf, sizeof(float) * CLASSES, cudaMemcpyDeviceToHost));
    printf("dL/df: \n");
    printMatrix(host_gradient, 1, CLASSES);

    printf("\nTotal time GPU (s): %f", tend - tstart);

    printf("\n");

    // free memory

    return 0;
}

void warmUpGPU() {
    printf("Warming up GPU for time trialing...");
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
