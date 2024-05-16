# CS453_DNN
Team 2 programmed a deep neural network from scratch using CUDA for their final project in CS453.

The network currently only runs on the CIFAR-10 Dataset. The network is composed of two fully connected layers with a reLU nonlinearity inbetween them and a softmax classifier. It is trained through stochastic gradient descent to predict which of the 10 classifications an input image belongs to. The entire network is allocated on the GPU in memory and minibatches of data are fed into the network. For example, on the K80 a minibatch of 1000 images can be passed in at once without exceeding it's memory capacity.

## To run:
There is a makefile in the /src directory that can be used to compile the program. I suggest using 'make remake' to build. The make file might need to be modified to build for the target architecture that you want to run on. It is currently only targeting one architecture for speed of compilation.
The program can be run by passing in the directory to the CIFAR-10 dataset. Pass in the path to the directory containing the .bin files, with no trailing /.

## To run on SLURM cluster:
This program was developed on NAU's SLURM cluster. There is a shell script called dnn.sh that can be used to make and run the program on slurm. This script will load cuda on the node, make the program, and submit an appropriate sbatch command and script to run on the cluster. Note that you do need to have the dataset copied to slurm for this to work. The filepath to the dataset is hardcoded in the shell script.

## Overview
The program works as follows:
1. The CIFAR 10 dataset is loaded into memory
2. The parameters for the network are all allocated on the GPU.
3. Initial values are generated for parameters, and they are copied to the GPU.
4. A randomly sampled minibatch of images is grabbed from the 'train' dataset and copied to the input of the network.
5. The forward api is invoked on all layers (which in turn invoke their own kernels).
6. The loss is calculated, and the gradient with respect to the loss.
7. The upstream gradient is backpropagated through the entire network through the backward api of each layer.
8. The gradient of the loss with respect to the parameters is then used to 'learn'/update the parameters. A momentum based approach is used to speed up convergence.
9. The process is repeated again for as many epochs are specified in the program.
10. Once training is complete, the accuracy of the network is cross validated on a validation set of 10,000 images.
11. Finally, a last set of 10,000 test images that were reserved can be used to determine the effectiveness of the model.
