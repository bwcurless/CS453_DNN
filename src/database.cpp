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
    int tempPush = 0, colors = 0, counter = 0;
    ifstream fin;

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
        outputData->yTrain.push_back(int(current[imageTrain * 3073]));

        for (colors = 0; colors < 3; colors++) {
            for (int pixelTrain = 0; pixelTrain < 1024; pixelTrain++) {
                //place pixels in x
                tempPush = pixelTrain + colors * 1024 + imageTrain * 3073 + 1;
                temp = current[tempPush];

                outputData->xTrain[pixelTrain + imageTrain * 1024][colors] = int(temp);
            }
        }
    }

    //place in struct, validation set
    for (int imageVal = (ELEM * trainPercent); imageVal < ELEM; imageVal++) {
        //place headers in y
        outputData->yVal.push_back(int(current[imageVal * 3073]));

        for (colors = 0; colors < 3; colors++) {
            for (int pixelVal = 0; pixelVal < 1024; pixelVal++) {
                //place pixels in x
                tempPush = pixelVal + colors * 1024 + imageVal * 3073 + 1;
                temp = current[tempPush];

                outputData->xVal[pixelVal + counter * 1024][colors] = int(temp);
            }
        }

        counter++;
    }

    return outputData;
}

//tests this function

int main() {
    cout << "Gathering data..." << endl;

    data_t *dataset = importDataset("data_batch_1.bin", 0.6);

    //for testing on https://onlinepngtools.com/convert-rgb-values-to-png
    cout << "First image pixels:" << endl;

    for (int indexRow = 0; indexRow < 32; indexRow++) {
        for (int indexCol = 0; indexCol < 32; indexCol++) {
            for (int inside = 0; inside < 3; inside++) {
                cout << " " << dataset->xTrain[indexRow * 32 + indexCol][inside];
            }
            cout << ";";
        }
        cout << endl;
    }

    return 0;
}
