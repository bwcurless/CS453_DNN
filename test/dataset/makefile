#Build test project
all: ./datasetTests.cpp ../../src/dataset.cu
	nvcc -std=c++17 -lstdc++fs -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp $^ -link -o datasetTests
