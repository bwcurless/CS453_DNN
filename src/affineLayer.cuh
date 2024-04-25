// A class that defines an affine layer for use in a neural network

#ifndef __AFFINELAYER_H__
#define __AFFINELAYER_H__

#include "params.h"

// The data required to compute an affine layer
typedef struct AffineInputs {
    float* W;
    float* x;
    float* b;
    float* f;
    float* dLdx;
    float* dLdW;
    float* dLdB;
    int numOutputs;
    int batchSize;
    int dataSize;
} affineInputs_t;

/*! \brief Initialize an affine layer
 *
 *  Allocate memory on device for this layer. You will need to deallocate the host and device memory
 * when you are done using this layer. Creates the data structure needed to use the affine layer.
 *
 * \param numOutputs Output size of this layer for a single input
 * \param batchSize The number of inputs in a batch
 * \param inputDataSize The length of a single input vector
 * \param x A minibatch of input data
 * \return affineInputs Passed to other layers
 */
affineInputs_t* affineInit(unsigned int numOutputs, unsigned int batchSize,
                           unsigned int inputDataSize, float* x);

/*! \brief Computes forward pass of Affine Layer
 *
 *  Computes the forward pass for affine layer. Just a matrix multiply of a batch of inputs 'x' with
 * a weight matrix 'W', with an offset 'b' applied. The results are stored in 'f'.
 *
 * \param  The inputs required to compute the forward pass of an affine layer.
 * \return void
 */
void affineForward(const affineInputs_t* inputs)
{
 unsigned int COL = threadIdx.x + blockDim.x * blockIdx.x;
 unsigned int ROW = threadIdx.y + blockDim.y * blockIdx.y;
 unsigned int localSum = 0;

 if (COL < inputs->dataSize && ROW < inputs->numOutputs)
 {
  for (unsigned int index = 0; index < inputs->dataSize; index++)
   {
    localSum += inputs->x[inputs->batchSize * index + COL] * inputs->W[ROW * inputs->dataSize + index];
   }
 }

 inputs->f[ROW * inputs->batchSize + COL] = localSum + inputs->b[ROW];

 return;
}

/*! \brief Computes backward pass of Affine Layer
 *
 *  Takes in cached values from forward pass, the upstream gradient, and computes the gradients of
 * the loss with respect to the inputs.
 *
 * \param upstreamGradient Upstream gradient of loss with respect to the output of this layer, f.
 * \param inputs The cached values used to compute the forward pass of this layer.
 * \return void
 */
void affineBackward(const float* upstreamGradient, const affineInputs_t* inputs);

// Performs gradient descent and updates the weights W and offsets b. Includes regularization for W
void affineUpdate(const learnParams_t* hyperParams, const affineInputs_t* inputs);


// 

#endif /* ifndef __AFFINELAYER_H__ */
