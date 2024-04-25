#Build test project
all: ./affineLayerTests.cu ../../src/affineLayer.cu
	nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp $^ -link -o affineLayerTests
