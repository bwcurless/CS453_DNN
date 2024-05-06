#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "dataset.h"

// Number of images per file
// This is constant for cifar-10
#define ELEM 1000
// Size of a cifar-10 image
#define DIM 1024

void readDataFile(char *data, const char *directory, std::filesystem::path filename);
void addToDataset(char *data, vector<vector<float> > *x, vector<uint8_t> *y);
/*
This function extracts information from a binary file
The input should be comprised of a multiple of 3073 bytes,
with 1 byte specifier and 32 x 32 bytes in order of Red,
Green, and Blue.
*/
data_t *importDataset(char *directoryPath) {
    printf("Importing dataset from directory: %s\n", directoryPath);
    // initialize variables
    data_t *outputData = new data_t;
    char *current = new char[ELEM * 3073];

    std::vector<float> inputData;

    // Import first few files as Train and validation dataset
    for (int fileNumber = 1; fileNumber < 3; fileNumber++) {
        // Test data
        std::filesystem::path filePath;
        filePath += "data_batch_";
        filePath += std::to_string(fileNumber);
        filePath += ".bin";
        readDataFile(current, directoryPath, filePath);
        // Only add one chunk to validation set
        if (fileNumber == 2) {
            addToDataset(current, &outputData->xVal, &outputData->yVal);
        } else {
            addToDataset(current, &outputData->xTrain, &outputData->yTrain);
        }
    }

    // Test data
    std::filesystem::path filePath;
    filePath = "test_batch.bin";
    readDataFile(current, directoryPath, filePath);
    addToDataset(current, &outputData->xTest, &outputData->yTest);

    delete[] current;
    return outputData;
}

void addToDataset(char *data, vector<vector<float> > *x, vector<uint8_t> *y) {
    std::vector<float> inputData;
    for (int image = 0; image < ELEM; image++) {
        // place headers in y
        y->push_back(uint8_t(data[image * 3073]));

        for (int pixel = 0; pixel < 3072; pixel++) {
            // place pixels in x
            int tempPush = pixel + image * 3073 + 1;
            unsigned char temp = data[tempPush];
            int tempInt = int(temp);

            // Build up the new vector
            inputData.push_back(float(tempInt));
        }
        // Add the new vector to our set
        x->push_back(inputData);
        inputData.clear();
    }
}

// Read in a file containing binary data. Load the data into the pre allocated buffer *data
void readDataFile(char *data, const char *directory, std::filesystem::path filename) {
    // Gather data and put in struct
    // open file for input
    std::filesystem::path filePath;
    filePath += directory;
    filePath /= filename;
    std::cout << "Opening file: " << filePath.string() << "\n";

    ifstream fin;
    fin.open(filePath, ios::in | ios::binary | ios::ate);

    if (!fin) {
        cout << "Error opening file \n";
        return;
    }

    // Place all data in char array
    fin.seekg(0, ios::beg);
    fin.read(data, ELEM * 3073);
    fin.close();
}
