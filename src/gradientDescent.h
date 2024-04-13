// Perform actual learning by gradient descent
#ifndef __GRADIENTDESCENT_H__
#define __GRADIENTDESCENT_H__
/*! \struct _learnParams_t
 *  \brief Hyper parameters for gradient descent
 *
 *  Contains all the hyper parameters for performing gradient descent. Uses a momentum based approach with exponential decay of old gradient
 */
typedef struct _learnParams_t {
	float learningRate; /*!< Gradient step size */
	float momentumDecay; /*!< Gradient decay for momentum */
} learnParams_t;


// Performs gradient descent and updates the weights. Includes regularization for fully connected layers as well
__global__ void gradientDescentRegularization(const learnParams_t *hyperParams, float regularizationStrength, float* weights, float* gradient, unsigned int numWeights);

// Performs gradient descent and updates the weights. Can be used for offsets b
__global__ void gradientDescent(const learnParams_t *hyperParams, float* weights, float* gradient, unsigned int numWeights);
#endif /* ifndef __GRADIENTDESCENT_H__ */
