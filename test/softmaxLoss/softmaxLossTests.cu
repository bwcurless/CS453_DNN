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

#define MINIBATCHSIZE 5
#define CLASSES 10

using namespace std;

// function prototypes
void warmUpGPU();
void compareMatrices(float *C_GPU, float *C_CPU, unsigned int NUMELEM);
template <typename T>
void printMatrix(T *matrix, int width, int height);
void outputSumElems(float *C, unsigned int NUMELEM);

int main(int argc, char *argv[]) {
    warmUpGPU();

    // seed random number generator with constant seed
    srand(123);

    // Set scores to small values, gaussian distribution, centered at 1, so we essentially have
    // equivalent scores
    float scale = 0.01;
    int scoresSize = sizeof(float) * CLASSES * MINIBATCHSIZE;
    float *host_scores = (float *)malloc(scoresSize);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(1.0, scale);

    for (int i = 0; i < CLASSES * MINIBATCHSIZE; i++) {
        host_scores[i] = distribution(generator);
    }
    printf("Scores:\n");
    printMatrix<float>(host_scores, MINIBATCHSIZE, CLASSES);

    // Push the scores to the GPU
    float *dev_f1;
    gpuErrchk(cudaMalloc((float **)&dev_f1, scoresSize));
    // Copy generated scores to device
    gpuErrchk(cudaMemcpy(dev_f1, host_scores, scoresSize, cudaMemcpyHostToDevice));

    // Allocate GPU space for softmax loss functions
    softmaxLoss_t *softmaxInputs = softmaxInit(CLASSES, MINIBATCHSIZE, dev_f1);

    // Init y values (actual classification)
    size_t ySize = sizeof(unsigned int) * MINIBATCHSIZE;
    unsigned int *host_y = (unsigned int *)malloc(ySize);
    for (int i = 0; i < MINIBATCHSIZE; i++) {
        host_y[i] = i % 10;
    }
    printf("Expected Classifications:\n");
    printMatrix<unsigned int>(host_y, MINIBATCHSIZE, 1);
    gpuErrchk(cudaMemcpy(softmaxInputs->y, host_y, ySize, cudaMemcpyHostToDevice));

    double tstart = omp_get_wtime();

    // execute kernel
    softmaxLoss(softmaxInputs);

    double tend = omp_get_wtime();

    // Copy Loss off GPU
    float host_loss;
    gpuErrchk(cudaMemcpy(&host_loss, softmaxInputs->loss, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Softmax Loss: %f\n", host_loss);

    // Copy Accuracy off GPU
    float host_accuracy;
    gpuErrchk(
        cudaMemcpy(&host_accuracy, softmaxInputs->accuracy, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Accuracy: %f\n", host_accuracy);

    // Copy gradient off GPU
    float *host_gradient;
    host_gradient = (float *)malloc(sizeof(float) * CLASSES * MINIBATCHSIZE);
    gpuErrchk(cudaMemcpy(host_gradient, softmaxInputs->dLdf,
                         sizeof(float) * CLASSES * MINIBATCHSIZE, cudaMemcpyDeviceToHost));
    printf("dL/df: \n");
    printMatrix<float>(host_gradient, MINIBATCHSIZE, CLASSES);

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

template <typename T>
void printMatrix(T *matrix, int width, int height) {
    int i, j;
    int cnt = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            printf("%.2f, ", (float)matrix[cnt]);
            cnt++;
        }
        printf("\n");
    }
}
