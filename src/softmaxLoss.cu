#include "cudaHelpers.cuh"
#include "softmaxLoss.cuh"

__global__ void normalizeSoftmaxOutputs(const softmaxLoss_t inputs);
__global__ void softmaxLossUnnormalized(const softmaxLoss_t inputs);

softmaxLoss_t *softmaxInit(unsigned int numClasses, unsigned int batchSize, float *f) {
    // The expected classes of the minibatch, used to train the model
    unsigned int *dev_y;
    gpuErrchk(cudaMalloc((unsigned int **)&dev_y, sizeof(unsigned int) * batchSize));

    // Softmax loss
    float *dev_softmax_loss;
    gpuErrchk(cudaMalloc((float **)&dev_softmax_loss, sizeof(float)));

    // Accuracy
    float *dev_accuracy;
    gpuErrchk(cudaMalloc((float **)&dev_accuracy, sizeof(float)));

    // Softmax dL/df. How much the loss changes with respect to each class score from the last layer
    float *dev_dLdf;
    gpuErrchk(cudaMalloc((float **)&dev_dLdf, sizeof(float) * numClasses * batchSize));

    // I guess this can leak, but don't feel like dealing with it now
    softmaxLoss_t *softmaxInputs = (softmaxLoss_t *)malloc(sizeof(softmaxLoss_t));
    softmaxInputs->loss = dev_softmax_loss;
    softmaxInputs->accuracy = dev_accuracy;
    softmaxInputs->dLdf = dev_dLdf;
    softmaxInputs->f = f;
    softmaxInputs->y = dev_y;
    softmaxInputs->numClasses = numClasses;
    softmaxInputs->batchSize = batchSize;

    return softmaxInputs;
}

void softmaxLoss(softmaxLoss_t *inputs) {
    // Init loss and accuracy to 0
    float zero = 0.0;
    cudaMemcpy(inputs->accuracy, &zero, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(inputs->loss, &zero, sizeof(float), cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());

    dim3 blockDim(128);
    dim3 gridDim(ceil(1.0 * inputs->batchSize / blockDim.x));
    size_t sharedSize = inputs->batchSize * inputs->numClasses * sizeof(float);
    // Dereference pointer because can't pass a reference to a struct to a kernel
    softmaxLossUnnormalized<<<gridDim, blockDim, sharedSize>>>(*inputs);
    gpuErrchk(cudaDeviceSynchronize());

    // Should be able to run this all in one block since numClasses is small
    dim3 blockDim2(32);
    dim3 gridDim2(1);
    normalizeSoftmaxOutputs<<<gridDim2, blockDim2>>>(*inputs);
    gpuErrchk(cudaDeviceSynchronize());

    // Compute regularization loss as well
    // It's not super important to have this, it's more important to factor it into the gradient
    // Im actually calculating this separately in the affine layer and adding it in later
}

// Normalize gradients and loss
__global__ void normalizeSoftmaxOutputs(const softmaxLoss_t inputs) {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid_x == 0) {
        *inputs.loss /= inputs.batchSize;
        *inputs.accuracy /= inputs.batchSize;
        // printf("Loss: %.3f\n", *inputs.loss);
        // printf("Training Accuracy: %.3f\n", *inputs.accuracy);
    }
}

__global__ void softmaxLossUnnormalized(const softmaxLoss_t inputs) {
    // Each thread is responsible for computing loss for one input
    // Should be geting in numClasses x batchSize scores
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float shared[];

    float *cachedScores = (float *)shared;

    // Compute loss
    if (tid_x < inputs.batchSize) {
        float imageLoss = 0.0;
        unsigned int correctClassIndex = inputs.y[tid_x];
        float correctClassScore = 0.0;
        float e_fSum = 0.0;

        // Find max score for this thread, cache values in shared memory so we don't repeat
        // global accesses
        float maxScore = 0.0;
        int maxScoreIndex = 0;
        for (int i = 0; i < inputs.numClasses; i++) {
            float val = inputs.f[i * inputs.batchSize + tid_x];
            cachedScores[threadIdx.x * inputs.numClasses + i] = val;
            if (i == 0) {
                maxScore = val;
                maxScoreIndex = i;
            } else if (val > maxScore) {
                maxScore = val;
                maxScoreIndex = i;
            }
        }

        if (maxScoreIndex == correctClassIndex) {
            atomicAdd(inputs.accuracy, 1);
        }

        // Subtract max score from all of them so we don't numerically blow up
        for (int i = 0; i < inputs.numClasses; i++) {
            cachedScores[threadIdx.x * inputs.numClasses + i] -= maxScore;
        }

        for (int i = 0; i < inputs.numClasses; i++) {
            // This is actually coalesced...
            float e_f = expf(cachedScores[threadIdx.x * inputs.numClasses + i]);
            // Cache the exponentiated score for computing gradient
            // We no longer need the original score, so can save memory and overwrite it with
            // the exponentiated score
            cachedScores[threadIdx.x * inputs.numClasses + i] = e_f;
            if (i == correctClassIndex) {
                correctClassScore = e_f;
            }
            e_fSum += e_f;
        }
        imageLoss = -logf(correctClassScore / e_fSum);

        // printf("Image %d loss is %f\n", tid_x, imageLoss);
        //   Each thread will reduce loss to here
        atomicAdd(inputs.loss, imageLoss);

        // Compute gradient dL/df
        // dL/df is computed for all images, and is (numClasses x batchSize)
        for (int i = 0; i < inputs.numClasses; i++) {
            float dLdf_i = 0.0;
            float softmax_i = cachedScores[threadIdx.x * inputs.numClasses + i] / e_fSum;
            if (i == correctClassIndex) {
                dLdf_i = -1 + softmax_i;
            } else {
                dLdf_i = softmax_i;
            }
            // Store gradients back out
            // tid_x is which column of the output this thread is responsible for
            // batchSize * i gets you to the correct row
            // Don't forget to normalize by number of inputs because the loss function is averaged
            // out over all batch samples
            inputs.dLdf[inputs.batchSize * i + tid_x] = dLdf_i / inputs.batchSize;
        }
    }
}
