// Imports a dataset and creates train/val/test splits
#ifndef __DATASET_H__
#define __DATASET_H__
#include <vector>

using namespace std;

typedef struct _data {
    vector<vector<char> > *xTrain;
    vector<char> *yTrain;
    vector<vector<char> > *xVal;
    vector<char> *yVal;
} data_t;

/*! \brief Import Dataset from file
 *
 *  Reads data from file and creates a train/val split of the data based on the requested percentages
 *
 * \param  filename path to dataset
 * \param  trainPercent Percentage of the dataset to use for training
 * \return data_t The Train/Val split dataset
 */
data_t * importDataset(const char *filename, float trainPercent);

#endif /* ifndef __DATASET_H__ */
