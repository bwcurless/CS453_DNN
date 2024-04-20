// To run this program, pass in the file path to the dataset and the neural network will train on
// the dataset, and then the accuracy will be evaluated.

#include <complex.h>
#include <cuda_runtime.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <random>
#include <vector>

#include "ReLULayer.cuh"
#include "affineLayer.cuh"
#include "cudaHelpers.cuh"
#include "dataset.h"
#include "params.h"
#include "softmaxLoss.cuh"

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

/* Function Prototypes */
void forward(affineInputs_t *aff1Inputs);

using namespace std;

int main(int argc, char *argv[]) {
    char filename[256];
    char *endptr;
    if (argc < 2) {
        fprintf(stderr, "Too few command line arguments. <pathToDataset> is required\n");
        return EXIT_FAILURE;
    }
    strcpy(filename, argv[1]);

    data_t *dataset = importDataset(filename, 0.6);
    // Should have a training dataset and a validation dataset. x are our inputs, y are the expected
    // outputs for each given input

    // ********* Construct the network *************
    // The network is essentially constructed from memory allocated to store all the data as it
    // propagates through the different layers, and the kernels that implement the different
    // layers, affine, ReLu, softmax, convolutional

    // Allocate memory for all intermediate steps on the GPU. This includes caching inputs to
    // each layer, outputs, and gradients used for backpropagation Input layer

    // Input layer
    float *dev_x;
    gpuErrchk(cudaMalloc((float **)&dev_x, sizeof(float) * MINIBATCHSIZE * INPUTSIZE));

    // First affine layer
    affineInputs_t *aff1Inputs = affineInit(CLASSES, MINIBATCHSIZE, INPUTSIZE, dev_x);

    // Softmax loss layer
    softmaxLoss_t *softmaxInputs = softmaxInit(CLASSES, MINIBATCHSIZE, aff1Inputs->f);

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
    gpuErrchk(cudaMemcpy(aff1Inputs->W, host_W1, W1Size, cudaMemcpyHostToDevice));

    // b1 needs to be set to 0 for no offsets at first
    int b1Size = sizeof(float) * CLASSES;
    float *host_b1 = (float *)malloc(b1Size);
    for (int i = 0; i < CLASSES; i++) {
        host_b1[i] = 0.0;
    }
    // Copy b1 to device
    gpuErrchk(cudaMemcpy(aff1Inputs->b, host_b1, b1Size, cudaMemcpyHostToDevice));

    // ******** Start of Optimization ************

    learnParams_t *learnParameters;
    learnParameters->learningRate = LEARNINGRATE;
    learnParameters->momentumDecay = MOMENTUMDECAY;
    learnParameters->regStrength = ALPHA;

    // Train for this many epochs
    for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
        // Iterate through as many minibatches as we need to complete an entire epoch
        for (int batch = 0; batch < ceil(1.0 * dataset->yTrain->size / MINIBATCHSIZE); batch++) {
            // Sample a minibatch of samples from training data
            unsigned int minibatchSize = MINIBATCHSIZE * INPUTSIZE;
            char *minibatch = (char *)malloc(sizeof(char) * minibatchSize);

            // TODO Sample the minibatch randomly from xTrain, and don't get any repeat inputs until
            // we are onto the next epoch

            // Push minibatch to GPU. Push images and expected classes
            gpuErrchk(
                cudaMemcpy(dev_x, minibatch, sizeof(char) * minibatchSize, cudaMemcpyHostToDevice));

            // Run forward and backward passes on minibatch of data, and update the gradient

            forward(aff1Inputs);
            // ReLU next
            // Another Affine layer

            // This layer computes the loss and the gradient of the loss with respect to the scores
            // input to this layer
            softmaxLoss(softmaxInputs);

            // At this point we will have the loss computed for every input image, and the gradient
            // of our softmax function. We now begin to backpropogate the gradients

            // Evaluate gradient for affine layer with respect to W and b f(x)=W*x+b, given the
            // upstream gradients and the last inputs
            affineBackward(softmaxInputs->dLdf, aff1Inputs);

            // Using our learning rate, update our parameters based on the gradient

            // Update Affine1 layer weights
            affineUpdate(learnParameters, aff1Inputs);

            // Print out the loss for debugging
            float loss;
            gpuErrchk(
                cudaMemcpy(&loss, softmaxInputs->loss, sizeof(float), cudaMemcpyDeviceToHost));
            printf("\nSoftmax Loss: %f", loss);
        }
    }

    // TODO Optional, save model off so we don't have to retrain in the future

    // Evaluate accuracy of classifier on training dataset
    float trainAccuracy;

    // TODO Run all the xTrain data through the model and evaluate the accuracy
    forward(aff1Inputs);

    printf("Train Accuracy: %f\n", trainAccuracy);

    // Evaluate accuracy of classifier on validation dataset
    float valAccuracy;

    // TODO Do the same for xVal and yVal and evaluate accuracy

    printf("Validation Accuracy: %f\n", valAccuracy);

    // Cleanup, free memory etc
}

/*! \brief Compute the forward pass
 *
 *  Used during training as well as for evaluating model performance. Evaluate forward pass for
 * entire network
 *
 * \param aff1Inputs Inputs for first affine layer
 * \return void
 */
void forward(affineInputs_t *aff1Inputs) {
    // Compute f(x)=W1*x+b1 forward pass
    affineForward(aff1Inputs);
}
