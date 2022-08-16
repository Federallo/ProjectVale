#ifndef _NEURALNETWORK_H
#define _NEURALNETWORK_H

//#include <eigen3/Eigen/Eigen>
#include <Eigen/Eigen>
#include <unsupported/Eigen/FFT>
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

class NeuralNetwork{
public:
    //TODO what is uint?
    NeuralNetwork(std::vector<uint> topology, float learningRate = float(0.005));

    //for forward propagandation of data
    //TODO what is RowVectorXf?
    void forwardPropagandation(Eigen::RowVectorXf& input);

    //for backward propagandation of errors made by neurons
    void backwardPropagandation(Eigen::RowVectorXf& output);

    //to determine errors made by neurons in each layer
    void calcError(Eigen::RowVectorXf& output);

    //to update weights of connections
    void updateWeights();

    //to train neural network give an array of data points
    void train(std::vector<std::unique_ptr<Eigen::RowVectorXf>> input_data, std::vector<std::unique_ptr<Eigen::RowVectorXf>> output_data);

private://for the moment
    //storage objects for working of neural network
    std::vector<std::unique_ptr<Eigen::RowVectorXf>> neuronLayers; //stores layers of out network
    std::vector<std::unique_ptr<Eigen::RowVectorXf>> cacheLayers; //stores the unactivated (activation functions not yet applied) values of layers
    std::vector<std::unique_ptr<Eigen::RowVectorXf>> deltas; //stores the error contribution of each neurons
    //TODO what is MatrixXf?
    std::vector<std::unique_ptr<Eigen::MatrixXf>> weights; //connection wights itself
    //TODO check
    std::vector<uint> topology;
    float learningRate;
};

#endif