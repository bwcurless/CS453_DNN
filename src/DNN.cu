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

#include <algorithm>
#include <cstdlib>
#include <iostream>
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
#define MINIBATCHSIZE 1000
#define NUMEPOCHS 20

// Hyper parameters
#define LEARNINGRATE 0.0001
#define ALPHA 25000
#define MOMENTUMDECAY 0.9

/* Function Prototypes */
void forward(affineInputs_t *aff1Inputs);
void printMatrix(float *matrix, int width, int height);
float randomRange(float min, float max);

using namespace std;

int main(int argc, char *argv[]) {
    char filename[256];
    char *endptr;
    if (argc < 2) {
        fprintf(stderr, "Too few command line arguments. <pathToDataset> is required\n");
        return EXIT_FAILURE;
    }
    strcpy(filename, argv[1]);

    data_t *dataset = importDataset(filename);
    printf("Dataset y size: %d\n", dataset->yTest.size());
    // Should have a training dataset and a validation dataset. x are our inputs, y are the expected
    // outputs for each given input

    // Normalization for training speed
    vector<float> means(dataset->xTrain[0].size(), 0.0);
    for (int image = 0; image < dataset->xTrain.size(); image++) {
        for (int pixel = 0; pixel < means.size(); pixel++) {
            means[pixel] += dataset->xTrain[image][pixel];
        }
    }
    // Calculate the mean
    for (int pixel = 0; pixel < means.size(); pixel++) {
        means[pixel] = means[pixel] / dataset->xTrain.size();
    }
    // Normalize to 0 mean
    for (int image = 0; image < dataset->xTrain.size(); image++) {
        for (int pixel = 0; pixel < means.size(); pixel++) {
            dataset->xTrain[image][pixel] -= means[pixel];
        }
    }

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

    int paramaterFindingIterations = 1;
    for (int iteration = 0; iteration < paramaterFindingIterations; iteration++) {
        float reg = 4e1;         //= pow(10.0, randomRange(4, 5));
        float learnRate = 5e-7;  //	pow(10.0, randomRange(-8, -4));
        printf("reg: %.1e, learn: %.1e\n", reg, learnRate);

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

        learnParams_t learnParameters;
        learnParameters.learningRate = learnRate;
        learnParameters.momentumDecay = MOMENTUMDECAY;
        learnParameters.regStrength = reg;

        double ttrainstart = omp_get_wtime();

        // keep track of best params
        // float bAcc, bLoss,

        // Train for this many epochs
        for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
            float runningAccuracy = 0.0;
            float runningLoss = 0.0;
            float runningRegLoss = 0.0;
            // Iterate through as many minibatches as we need to complete an entire epoch
            int numBatches = ceil(1.0 * dataset->yTrain.size() / MINIBATCHSIZE);
            for (int batch = 0; batch < numBatches; batch++) {
                // printf("Epoch: %d, Minibatch (%d/%d)\n", epoch, batch, numBatches);
                //  Sample a minibatch of samples from training data
                unsigned int minibatchXSize = sizeof(float) * MINIBATCHSIZE * INPUTSIZE;
                float *minibatchX = (float *)malloc(minibatchXSize);
                unsigned int minibatchYSize = sizeof(unsigned int) * MINIBATCHSIZE;
                unsigned int *minibatchY = (unsigned int *)malloc(minibatchYSize);

                // Sample the minibatch randomly from xTrain, and don't get any repeat inputs until
                // we are onto the next epoch
                // Generate series of random numbers up to minibatchSize and access those indexes
                // sequentially from there
                std::vector<unsigned int> indices(dataset->yTrain.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::random_shuffle(indices.begin(), indices.end());

                // Compose a new randomly sampled minibatch
                for (int i = 0; i < MINIBATCHSIZE; i++) {
                    int indice = batch * MINIBATCHSIZE + i;
                    // The last minibatch will have empty slots to fill with input data, wrap back
                    // around for simplicity
                    if (indice >= dataset->yTrain.size()) {
                        indice = indice % dataset->yTrain.size();
                    }
                    int randomIndice = indices[indice];
                    // printf("Random Indice: %d\n", randomIndice);
                    //  Copy over the entire vector
                    for (int dim = 0; dim < INPUTSIZE; dim++) {
                        // Need to push it on in a transposed fashion. Can't just push it row by
                        // row, because the matrix x is really transposed from that orientation.
                        minibatchX[MINIBATCHSIZE * dim + i] = dataset->xTrain[randomIndice][dim];
                    }
                    minibatchY[i] = dataset->yTrain[randomIndice];
                }

                // Push minibatch to GPU. Push images and expected classes
                gpuErrchk(cudaMemcpy(dev_x, minibatchX, minibatchXSize, cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy(softmaxInputs->y, minibatchY, minibatchYSize,
                                     cudaMemcpyHostToDevice));
                free(minibatchX);
                free(minibatchY);

                // Run forward and backward passes on minibatch of data, and update the gradient

                forward(aff1Inputs);
                // ReLU next
                // Another Affine layer

                // This layer computes the loss and the gradient of the loss with respect to the
                // scores input to this layer
                softmaxLoss(softmaxInputs);

                // At this point we will have the loss computed for every input image, and the
                // gradient of our softmax function. We now begin to backpropogate the gradients

                // Evaluate gradient for affine layer with respect to W and b f(x)=W*x+b, given the
                // upstream gradients and the last inputs
                affineBackward(learnParameters.regStrength, softmaxInputs->dLdf, aff1Inputs);

                // Using our learning rate, update our parameters based on the gradient

                // Update Affine1 layer weights
                affineUpdate(&learnParameters, aff1Inputs);

                gpuErrchk(cudaMemcpy(host_b1, aff1Inputs->b, b1Size, cudaMemcpyDeviceToHost));
                // printf("b\n");
                // for (int i = 0; i < CLASSES; i++) {
                //     printf("%f, ", host_b1[i]);
                // }
                // printf("\nExpected Classes were:\n");
                // for (int i = 0; i < MINIBATCHSIZE; i++) {
                //    printf("%d, ", minibatchY[i]);
                //}
                // Copy f
                // float *host_f = (float *)malloc(sizeof(float) * MINIBATCHSIZE * CLASSES);
                // gpuErrchk(cudaMemcpy(host_f, aff1Inputs->f, sizeof(float) * MINIBATCHSIZE *
                // CLASSES,
                //                     cudaMemcpyDeviceToHost));
                // printf("\nf\n");
                // printMatrix(host_f, MINIBATCHSIZE, CLASSES);

                // Pull accuracy
                float softmaxLoss;
                gpuErrchk(cudaMemcpy(&softmaxLoss, softmaxInputs->loss, sizeof(float),
                                     cudaMemcpyDeviceToHost));
                float regLoss;
                gpuErrchk(cudaMemcpy(&regLoss, aff1Inputs->regLoss, sizeof(float),
                                     cudaMemcpyDeviceToHost));
                float accuracy;
                gpuErrchk(cudaMemcpy(&accuracy, softmaxInputs->accuracy, sizeof(float),
                                     cudaMemcpyDeviceToHost));
                runningAccuracy += accuracy;
                runningLoss += softmaxLoss + regLoss;
                runningRegLoss += regLoss;
            }
            runningAccuracy = runningAccuracy / numBatches;

            printf("Averaged Accuracy: %f\n", runningAccuracy);
            printf("Averaged Loss: %f\n", runningLoss);
            printf("Regularization Loss: %f\n", runningRegLoss);
        }
        double ttrainend = omp_get_wtime();
        printf("Training Time: %f\n", ttrainend - ttrainstart);
    }

    // Evaluate accuracy of classifier on training dataset
    float trainAccuracy;

    // TODO Run all the xTrain data through the model and evaluate the accuracy
    forward(aff1Inputs);

    printf("\nFinal Results:\n");
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

void printMatrix(float *matrix, int width, int height) {
    int i, j;
    int cnt = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            printf("%+.2f, ", matrix[cnt]);
            cnt++;
        }
        printf("\n");
    }
}

float randomRange(float min, float max) {
    int range = max - min;
    // Random number [0-1)
    float num = rand() / (RAND_MAX + 1.);
    num = num * range + min;
    printf("Random Number is: %.2f\n", num);
    return num;
}
