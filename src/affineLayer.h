// A class that defines an affine layer for use in a neural network

#ifndef __AFFINELAYER_H__
#define __AFFINELAYER_H__

// The inputs required for the forward pass
typedef struct AffineInputs{
	float* W;
	float* x;
	float* b;
	float* f;
	int numOutputs;
	int batchSize;
	int dataSize;
} AffineInputs;

// The gradients that are output from this layer during backward pass
typedef struct AffineGradients{
	float* dLdW;
	float* dLdB;
} AffineGradients;

/*! \brief Computes forward pass of Affine Layer
 *
 *  Computes the forward pass for affine layer. Just a matrix multiply of a batch of inputs 'x' with a weight matrix 'W', with an offset 'b' applied. The results are stored in 'f'.
 *
 * \param  The inputs required to compute the forward pass of an affine layer.
 * \return void
 */
__global__ void affineForward(const AffineInputs *inputs);

/*! \brief Computes backward pass of Affine Layer
 *
 *  Takes in cached values from forward pass, the upstream gradient, and computes the gradients of the loss with respect to the inputs.
 *
 * \param dLdf The upstream gradient of loss with respect to the output of this layer, f.
 * \param inputs The cached values used to compute the forward pass of this layer.
 * \param gradients The gradients that this backward pass computes will be stored here.
 * \return void
 */
__global__ void affineBackward(const float* dLdf, const AffineInputs *inputs, const AffineGradients *gradients);

// Performs gradient descent and updates the weights W and offsets b. Includes regularization for W
__global__ void affineUpdate(const learnParams_t *hyperParams, const AffineInputs *inputs, const AffineGradients *gradients);

#endif /* ifndef __AFFINELAYER_H__ */
