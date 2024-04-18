// A rectified linear unit activation function to be used as a layer in a neural network.
#ifndef __RELULAYER_H__
#define __RELULAYER_H__

typedef struct _reluInputs {
    float *inputs;    /*!< Inputs to the ReLU layer */
    float *outputs;   /*!< Outputs from the ReLU layer */
    unsigned int dim; /*!< Number of elements in the input and output */
} reluInput_t;

// outputs are max(0, input). Essentially we threshold everything negative to 0, and we pass
// everything > 0 straight through unaltered.
__global__ void reluForward(reluInput_t *inputs);

// gradientsOut is 1 * upstreamGradients if the input for that gradient was > 0, and gradient is 0
// if the input for that gradient was <= 0.
__global__ void reluBackward(float *upstreamGradients, reluInput_t *inputs, float *gradientsOut);

#endif /* ifndef __RELULAYER_H__ */
