// A rectified linear unit activation function to be used as a layer in a neural network.
#ifndef __RELULAYER_H__
#define __RELULAYER_H__

typedef struct _reluInputs {
	float *inputs;/*!< Inputs to the ReLU layer */
	float *outputs;/*!< Outputs from the ReLU layer */
	unsigned int dim;/*!< Number of elements in the input and output */
} reluInput_t;

void reluForward(reluInput_t inputs);

void reluBackward(reluInput_t inputs, float *gradients);

#endif /* ifndef __RELULAYER_H__ */
