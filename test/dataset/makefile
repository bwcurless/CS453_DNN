#Build test project
all: ./datasetTests.cpp ../../src/dataset.cpp
	nvcc -O3 -arch=compute_80 -code=sm_80 -lcuda -Xcompiler -fopenmp $^ -link -o datasetTests
