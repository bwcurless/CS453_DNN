// Computes Softmax loss and gradient. Used as the last layer in our neural network

#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

typedef struct _softmaxLoss_t {
    float* f;                 // Input predictions
    unsigned int* y;          // Expected classes
    unsigned int numClasses;  // How many classes there are
    unsigned int batchSize;   // How many inputs were processed
    float* loss;              // How much loss was incurred over all classes
    float* dLdf;              // Gradient of loss with respect to predictions
} softmaxLoss_t;

/*! \brief Compute the softmax loss and gradient for the given input predictions
 *
 *  Pass in predictions, expected classes, number of classes, batch size to compute the softmax loss
 * of the batch as well as the gradient of the loss with respect to the input predictions.
 *
 * \param  inputs Struct required to compute the softmax loss
 * \return void
 */
void softmaxLoss(softmaxLoss_t* inputs);

/*! \brief Initialize softmax laye
 *
 *  Allocate space on GPU, creates data structure for computing softmax loss.
 *
 * \param numClasses Number of classes to compute loss over
 * \param batchSize Number of samples per batch
 * \param f Input scores, previously allocated
 * \return Data structure
 */
softmaxLoss_t* softmaxInit(unsigned int numClasses, unsigned int batchSize, float* f);

#endif /* ifndef __SOFTMAX_H__ */
