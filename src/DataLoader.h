#ifndef _DATALOADER_H
#define _DATALOADER_H

#include <Eigen/Eigen>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <iostream>
#include <memory>
#include <fstream>

class DataLoader
{
public:
    //DataLoader();

    //method to read csv file
    void readCSV(std::string filename, std::vector<std::unique_ptr<Eigen::RowVectorXf>>& data);

    //to generate training data
    void generateData(std::string filename);
};

#endif