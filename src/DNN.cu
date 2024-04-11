// To run this program, pass in the file path to the dataset and the neural network will train on
// the dataset, and then the accuracy will be evaluated.

#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> /* time */
#include <unistd.h>

#include <iostream>
#include <random>

// Run everything on CPU
// If not defined, will run GPU implementation
#define CPU

// Number of classes to predict for on output layer
#define CLASSES 10
// Size of the NN input layer (The size of the flattened image)
#define INPUTSIZE 3072
#define TRAINSIZE 10000
#define MINIBATCHSIZE 1000
#define NUMEPOCHS 100

// Hyper parameters
#define LEARNINGRATE 0.001
#define ALPHA 0.00001

// Error checking GPU calls
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using namespace std;

int main(int argc, char *argv[]) {
    // Do some argument parsing
    // Parse out the filename

    // Read in dataset from file
    // Should have a training dataset, a validation dataset, and a test dataset

    // ********* Construct the network *************
    // The network is essentially constructed from memory allocated to store all the data as it
    // propagates through the different layers, and the kernels that implement the different layers,
    // affine, ReLu, softmax, convolutional

    // Allocate memory for all intermediate steps on the GPU. This includes caching inputs to each
    // layer, outputs, and gradients used for backpropagation Input layer
    float *dev_inputLayer;
    int inputLayerSize = sizeof(float) * MINIBATCHSIZE;
    gpuErrchk(cudaMalloc((float **)&dev_inputLayer, inputLayerSize));

    // W1. The weight matrix we are trying to find
    float *dev_W1;
    gpuErrchk(cudaMalloc((float **)&dev_W1, sizeof(float) * CLASSES * INPUTSIZE));

    // dL/dW1. How much the weights effect the loss
    float *dev_dLdW1;
    gpuErrchk(cudaMalloc((float **)&dev_dLdW1, sizeof(float) * CLASSES * INPUTSIZE));

    // b1. The biases for each output of the linear classifier. The +b term
    float *dev_b1;
    gpuErrchk(cudaMalloc((float **)&dev_b1, sizeof(float) * CLASSES));

    // dL/dB1. How much the biases effect the loss
    float *dev_dldB1;
    gpuErrchk(cudaMalloc((float **)&dev_dldB1, sizeof(float) * CLASSES));

    // Intermediate Scores f(x). The linear classifier's predicted scores f(x)=W*x+b
    float *dev_f1;
    gpuErrchk(cudaMalloc((float **)&dev_f1, sizeof(float) * CLASSES));

    // Softmax scores (Probabilities input is of a certain class)
    float *dev_softmax_scores;
    gpuErrchk(cudaMalloc((float **)&dev_softmax_scores, sizeof(float) * CLASSES));

    // Softmax loss
    float *dev_loss;
    gpuErrchk(cudaMalloc((float **)&dev_loss, sizeof(float)));

    // Softmax dL/df1. How much the loss changes with respect to each class score
    float *dev_dLdf1;
    gpuErrchk(cudaMalloc((float **)&dev_dLdf1, sizeof(float) * CLASSES));

    // ****** Initialize Model Parameters *********

    // W1 needs to be set to small values, gaussian distribution, 0 mean
    float weightScale = 0.001;
    int W1Size = sizeof(float) * CLASSES * INPUTSIZE;
    float *host_W1 = (float *)malloc(W1Size);
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, weightScale);

    for (int i = 0; i < CLASSES * INPUTSIZE; i++) {
        host_W1[i] = distribution(generator);
    }
    // Copy W1 to device
    gpuErrchk(cudaMemcpy(dev_W1, host_W1, W1Size, cudaMemcpyHostToDevice));

    // b1 needs to be set to 0 for no offsets at first
    int b1Size = sizeof(float) * CLASSES;
    float *host_b1 = (float *)malloc(b1Size);
    for (int i = 0; i < CLASSES; i++) {
        host_b1[i] = 0.0;
    }
    // Copy b1 to device
    gpuErrchk(cudaMemcpy(dev_b1, host_b1, b1Size, cudaMemcpyHostToDevice));

    // ******** Start of Optimization ************

    // Descend gradient for this many epochs
    for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
        // Iterate through as many minibatches as we need to complete an entire epoch
        for (int batch = 0; batch < ceil(1.0 * TRAINSIZE / MINIBATCHSIZE); batch++) {
            // Push minibatch to GPU
            // gpuErrchk(cudaMemcpy(dev_inputLayer, &trainData[minibatchStartIndex], sizeof(float) *
            // MINIBATCHSIZE);

            // Perform gradient descent here
            // TODO Optional perform Stochastic gradient descent and iterate on a minibatch of
            // inputs to speed up convergence

            // Take all images in training dataset and pass them through the network
            // Each layer will cache it's gradient in the pre allocated memory space as we go to
            // prepare for backpropogation

            // At this point we will have the loss computed for every input image, and the gradient
            // of our softmax function. We now begin to backpropogate the gradients Backpropogate
            // the gradient with respect to all parameters all the way through the network

            // Using our learning rate, update our parameters based on the gradient
        }
    }

    // TODO Optional, save model off so we don't have to retrain in the future

    // Evaluate accuracy of classifier on training dataset

    // Evaluate accuracy of classifier on validation dataset

    // Evaluate accuracy of classifier on test dataset (Don't really need to do this since this is
    // mostly just for fun)

    // Cleanup, free memory etc
}
