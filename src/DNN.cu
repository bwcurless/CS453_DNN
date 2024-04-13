// To run this program, pass in the file path to the dataset and the neural network will train on
// the dataset, and then the accuracy will be evaluated.

#include <complex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> /* time */
#include <unistd.h>

#include <iostream>
#include <random>

#include "affineLayer.h"
#include "softmaxLoss.h"

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
#define MOMENTUMDECAY 0.75

/*! \struct _learnParams_t
 *  \brief Hyper parameters for gradient descent
 *
 *  Contains all the hyper parameters for performing gradient descent. Uses a momentum based
 * approach with exponential decay of old gradient
 */
typedef struct _learnParams_t {
    float learningRate;  /*!< Gradient step size */
    float momentumDecay; /*!< Gradient decay for momentum */
    float regStrength;   /*!< Regularization strength for fully connected layers */
} learnParams_t;

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
    float *dev_x1;
    gpuErrchk(cudaMalloc((float **)&dev_x1, sizeof(float) * MINIBATCHSIZE * INPUTSIZE));

    // W1. The weight matrix we are trying to find
    float *dev_W1;
    gpuErrchk(cudaMalloc((float **)&dev_W1, sizeof(float) * CLASSES * INPUTSIZE));

    // b1. The biases for each output of the linear classifier. The +b term
    float *dev_b1;
    gpuErrchk(cudaMalloc((float **)&dev_b1, sizeof(float) * CLASSES));

    // Intermediate Scores f(x). The linear classifier's predicted scores f(x)=W*x+b
    float *dev_f1;
    gpuErrchk(cudaMalloc((float **)&dev_f1, sizeof(float) * CLASSES));

    AffineInputs *aff1Inputs;
    aff1Inputs->W = dev_W1;
    aff1Inputs->x = dev_x1;
    aff1Inputs->b = dev_b1;
    aff1Inputs->f = dev_f1;
    aff1Inputs->batchSize = MINIBATCHSIZE;
    aff1Inputs->dataSize = INPUTSIZE;
    aff1Inputs->numOutputs = CLASSES;

    // dL/dW1. How much the weights effect the loss
    float *dev_dLdW1;
    gpuErrchk(cudaMalloc((float **)&dev_dLdW1, sizeof(float) * CLASSES * INPUTSIZE));

    // dL/db1. How much the biases effect the loss
    float *dev_dLdb1;
    gpuErrchk(cudaMalloc((float **)&dev_dLdb1, sizeof(float) * CLASSES));

    AffineGradients *aff1Grads;
    aff1Grads->dLdB = dev_dLdb1;
    aff1Grads->dLdW = dev_dLdW1;

    // The expected classes of the minibatch, used to train the model
    float *dev_y;
    gpuErrchk(cudaMalloc((float **)&dev_y, sizeof(float) * MINIBATCHSIZE));

    // Softmax loss
    float *dev_softmax_loss;
    gpuErrchk(cudaMalloc((float **)&dev_softmax_loss, sizeof(float)));

    // Softmax dL/df. How much the loss changes with respect to each class score from the last layer
    float *dev_dLdf;
    gpuErrchk(cudaMalloc((float **)&dev_dLdf, sizeof(float) * CLASSES));

    softmaxLoss_t *softmaxInputs;
    softmaxInputs->loss = dev_softmax_loss;
    softmaxInputs->dLdf = dev_dLdf;
    softmaxInputs->f = dev_f1;
    softmaxInputs->numClasses = CLASSES;
    softmaxInputs->batchSize = MINIBATCHSIZE;

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

    learnParams_t *learnParameters;
    learnParameters->learningRate = LEARNINGRATE;
    learnParameters->momentumDecay = MOMENTUMDECAY;
    learnParameters->regStrength = ALPHA;

    // Descend gradient for this many epochs
    for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
        // Iterate through as many minibatches as we need to complete an entire epoch
        for (int batch = 0; batch < ceil(1.0 * TRAINSIZE / MINIBATCHSIZE); batch++) {
            // Sample a minibatch of samples from training data

            // Push minibatch to GPU. Push images and expected classes
            // gpuErrchk(cudaMemcpy(dev_inputLayer, &trainData[minibatchStartIndex], sizeof(float) *
            // MINIBATCHSIZE);

            // Run forward and backward passes on minibatch of data, and update the gradient

            // Compute f(x)=W1*x+b1 forward pass
            dim3 blockDim(32, 32);
            // Number of threads is the size of the output matrix
            dim3 gridDim(ceil(1.0 * MINIBATCHSIZE / blockDim.x), ceil(1.0 * CLASSES / blockDim.y));
            affineForward<<<gridDim, blockDim>>>(affine1Inputs);

            // This layer computes the loss and the gradient of the loss with respect to the scores
            // input to this layer
            dim3 blockDim(32, 32);
            // Number of threads is the size of the output matrix of scores
            dim3 gridDim(ceil(1.0 * MINIBATCHSIZE / blockDim.x), ceil(1.0 * CLASSES / blockDim.y));
            softmaxLoss<<<gridDim, blockDim>>>(softmaxInputs);

            // At this point we will have the loss computed for every input image, and the gradient
            // of our softmax function. We now begin to backpropogate the gradients

            // Evaluate gradient for affine layer with respect to W and b f(x)=W*x+b, given the
            // upstream gradients and the last inputs
            dim3 blockDim(32, 32);
            // Number of threads is the size of the output matrix
            dim3 gridDim(ceil(1.0 * MINIBATCHSIZE / blockDim.x), ceil(1.0 * CLASSES / blockDim.y));
            affineBackward<<<gridDim, blockDim>>>(dev_dLdf, aff1Inputs, aff1Grads);

            // Using our learning rate, update our parameters based on the gradient

            // Update Affine1 layer weights
            dim3 blockDim(32, 32);
            dim3 gridDim(ceil(1.0 * MINIBATCHSIZE / blockDim.x), ceil(1.0 * CLASSES / blockDim.y));
            affineUpdate<<<gridDim, blockDim>>>(learnParameters, aff1Inputs, aff1Grads);

            // Print out the loss for debugging
            float loss;
            gpuErrchk(cudaMemcpy(&loss, dev_softmax_loss, sizeof(float), cudaMemcpyDeviceToHost));
            printf("\nSoftmax Loss: %f", loss);
        }
    }

    // TODO Optional, save model off so we don't have to retrain in the future

    // Evaluate accuracy of classifier on training dataset

    // Evaluate accuracy of classifier on validation dataset

    // Evaluate accuracy of classifier on test dataset (Don't really need to do this since this is
    // mostly just for fun)

    // Cleanup, free memory etc
}
