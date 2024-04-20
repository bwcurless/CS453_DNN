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

#define MINIBATCHSIZE 1000
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
    // printMatrix(host_scores, MINIBATCHSIZE, CLASSES);

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
        // host_y[i] = i % 10;  // This provides 0 gradient, essentially we guessed in such a
        //  way that the gradients cancel out across the board. I think a similar thing would happen
        //  if we initialized all the weights of our matrix to 0, or to a constant value I think.
        //  Essentially if there is no asymmetry to a minibatch you'll get no gradient since all the
        //  gradients will average out to be 0 across the batch.
        host_y[i] = 1;
    }
    gpuErrchk(cudaMemcpy(softmaxInputs->y, host_y, ySize, cudaMemcpyHostToDevice));

    double tstart = omp_get_wtime();

    // execute kernel
    softmaxLoss(softmaxInputs);

    double tend = omp_get_wtime();

    // Copy Loss off GPU
    float *host_loss;
    host_loss = (float *)malloc(sizeof(float));
    gpuErrchk(cudaMemcpy(host_loss, softmaxInputs->loss, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Softmax Loss: %f\n", *host_loss);

    // Copy gradient off GPU
    float *host_gradient;
    host_gradient = (float *)malloc(sizeof(float) * CLASSES);
    gpuErrchk(cudaMemcpy(host_gradient, softmaxInputs->dLdf, sizeof(float) * CLASSES,
                         cudaMemcpyDeviceToHost));
    printf("dL/df: \n");
    printMatrix(host_gradient, 1, CLASSES);

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
