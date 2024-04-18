#include "softmaxLoss.cuh"

__global__ void softmaxLoss(const softmaxLoss_t *inputs) {
    // Each thread is responsible for computing one loss
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
    }

    // Compute gradient dL/df
}
