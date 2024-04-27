#include "dataset.h"

/*
This function extracts information from a binary file
The input should be comprised of a multiple of 3073 bytes,
with 1 byte specifier and 32 x 32 bytes in order of Red,
Green, and Blue.
*/
data_t * importDataset(const char *filename, float trainPercent) {
    //initialize variables
    data_t * outputData = new data_t;
    char * current = new char[ELEM * 3073];
    unsigned char temp;
    int tempPush = 0, counter = 0, tempInt = 0;
    ifstream fin;

    std::vector<float> inputData;

    //Gather data and put in struct
    //open file for input
    fin.open(filename,ios::in | ios::binary | ios::ate);

    if (!fin) {
        cout << "Error opening file \n";
        return outputData;
    }

    //Place all data in char array
    fin.seekg(0, ios::beg);
    fin.read(current, ELEM * 3073);
    fin.close();

    //place in struct, training set
    for (int imageTrain = 0; imageTrain < ELEM * trainPercent - 1; imageTrain++) {
        //place headers in y
        outputData->yTrain.push_back(uint8_t(current[imageTrain * 3073]));

        for (int pixelTrain = 0; pixelTrain < 3072; pixelTrain++) {
            //place pixels in x
            tempPush = pixelTrain + imageTrain * 3073 + 1;
            temp = current[tempPush];
            tempInt = int(temp);

	    // Build up the new vector
            inputData.push_back(float(tempInt));
        }
	// Add the new vector to our set
	outputData->xTrain.push_back(inputData);
	inputData.clear();
    }

    //place in struct, validation set
    for (int imageVal = (ELEM * trainPercent); imageVal < ELEM; imageVal++) {
        //place headers in y
        outputData->yVal.push_back(uint8_t(current[imageVal * 3073]));

        for (int pixelVal = 0; pixelVal < 3072; pixelVal++) {
             //place pixels in x
            tempPush = pixelVal + imageVal * 3073 + 1;
            temp = current[tempPush];
            tempInt = int(temp);

	    // Build up the new vector
            inputData.push_back(float(tempInt));
        }
	// Add the new vector to our set
	outputData->xVal.push_back(inputData);
	inputData.clear();

        counter++;
    }

    delete current;
    return outputData;
}
