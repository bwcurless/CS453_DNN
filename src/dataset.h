// Imports a dataset and creates train/val/test splits
#ifndef __DATASET_H__
#define __DATASET_H__
#include <vector>
#include <cstdint>


using namespace std;

typedef struct _data {
    vector<vector<float> > xTrain;
    vector<uint8_t> yTrain;
    vector<vector<float> > xVal;
    vector<uint8_t> yVal;
    vector<vector<float> > xTest;
    vector<uint8_t> yTest;
} data_t;

/*! \brief Import Dataset from file
 *
 *  Reads data from file and creates a train/val split of the data based on the requested percentages
 *
 * \param  filename path to dataset
 * \param  trainPercent Percentage of the dataset to use for training
 * \return data_t The Train/Val split dataset
 */
data_t *importDataset(char *directoryPath);

#endif /* ifndef __DATASET_H__ */
