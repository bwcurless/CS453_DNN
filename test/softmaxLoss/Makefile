#Build test project
all: ./softmaxLossTests.cu ../../src/softmaxLoss.cu
	nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp $^ -link -o softmaxLossTests
