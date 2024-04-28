#Build test project
all: ./ReLULayerTests.cu ../../src/ReLULayer.cu
	nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp $^ -link -o ReLULayerTests
