//including directories
#include <Eigen/Eigen>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <memory>
#include "DataLoader.h"
#include "NeuralNetwork.h"

int main()
{
    //code here
    NeuralNetwork n({2, 3, 1});
    DataLoader loader;
    std::vector<std::unique_ptr<Eigen::RowVectorXf>> inputData, outputData;
    loader.generateData("test");
    loader.readCSV("test-in", inputData);
    loader.readCSV("test-out", outputData);
    n.train(inputData, outputData);
    return 0;
}