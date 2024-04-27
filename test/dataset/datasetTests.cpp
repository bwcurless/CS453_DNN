#include "../../src/dataset.h"

int main(int argc, char *argv[]) {
    cout << "Gathering data..." << endl;

    data_t *dataset = importDataset(argv[1], 0.6);

    printf("xTrain Number of Vectors: %ld\n", dataset->xTrain.size());
    printf("xTrain Vector Length: %ld\n", dataset->xTrain[0].size());
    printf("yTrain Length: %ld\n", dataset->yTrain.size());
    printf("xVal Numer of Vectors: %ld\n", dataset->xVal.size());
    printf("xVal Vector Length: %ld\n", dataset->xVal[0].size());
    printf("yVal Length: %ld\n", dataset->yVal.size());

    //for testing on https://onlinepngtools.com/convert-rgb-values-to-png
    cout << "First image pixels:" << endl;

    for (int indexRow = 0; indexRow < 32; indexRow++) {
        for (int indexCol = 0; indexCol < 32; indexCol++) {
		for (int color = 0; color < 3; color++) {
            cout << " " << dataset->xTrain[0][indexRow * 32 + indexCol];
		}
        }
        cout << endl;
    }

    return 0;
}
