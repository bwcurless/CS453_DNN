#include "softmaxLoss.cuh"

__global__ void softmaxLoss(const softmaxLoss_t *inputs) {
    // Each thread is responsible for computing loss for one input
    // Should be geting in numClasses x batchSize scores
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float shared[];

    float *cachedScores = (float *)shared;

    // Compute loss
    if (tid_x < inputs->batchSize) {
        float imageLoss = 0.0;
        unsigned int correctClassIndex = inputs->y[tid_x];
        float correctClassScore = 0.0;
        float e_fSum = 0.0;

        // Find max score for this thread, cache values in shared memory so we don't repeat global
        // accesses
        float maxScore = 0.0;
        for (int i = 0; i < inputs->numClasses; i++) {
            float val = inputs->f[i * inputs->batchSize + tid_x];
            cachedScores[threadIdx.x * inputs->numClasses + i] = val;
            if (i == 0) {
                maxScore = val;
            } else if (val > maxScore) {
                maxScore = val;
            }
        }

        // Subtract max score from all of them so we don't numerically blow up
        for (int i = 0; i < inputs->numClasses; i++) {
            cachedScores[threadIdx.x * inputs->numClasses + i] -= maxScore;
        }

        for (int i = 0; i < inputs->numClasses; i++) {
            // This is actually coalesced...
            float e_f = expf(cachedScores[threadIdx.x * inputs->numClasses + i]);
            // Cache the exponentiated score for computing gradient
            // We no longer need the original score, so can save memory and overwrite it with the
            // exponentiated score
            cachedScores[threadIdx.x * inputs->numClasses + i] = e_f;
            if (i == correctClassIndex) {
                correctClassScore = e_f;
            }
            e_fSum += e_f;
        }
        imageLoss = -logf(correctClassScore / e_fSum);
        // Normalize loss since we have an entire batch, easier to do here than sync entire grid or
        // run another kernel
        imageLoss /= inputs->batchSize;

        // Each thread will reduce exponentiated score to here
        atomicAdd(inputs->loss, imageLoss);

        // Compute gradient dL/df
        // dL/df is the average of all images gradients, and is (numClasses x batchSize)
        for (int i = 0; i < inputs->numClasses; i++) {
            float dLdf_i = 0.0;
            float softmax_i = cachedScores[threadIdx.x * inputs->numClasses + i] / e_fSum;
            if (i == correctClassIndex) {
                dLdf_i = -1 + softmax_i;
            } else {
                dLdf_i = softmax_i;
            }
            // Normalize, should really do this just once after entire grid finishes though
            dLdf_i /= inputs->batchSize;
            // Average gradient will be reduced to ohere
            atomicAdd(&(inputs->dLdf[tid_x * inputs->numClasses + i]), dLdf_i);
        }
    }
}
