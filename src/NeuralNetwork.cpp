#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<uint> topology, float learningRate): topology(topology), learningRate(learningRate)
{
    for(uint i = 0; i < topology.size(); i ++)
    {
        //neuron layers initialization
        if(i == topology.size() - 1)
            neuronLayers.push_back(std::unique_ptr<Eigen::RowVectorXf>(new Eigen::RowVectorXf(topology[i])));
        else
            neuronLayers.push_back(std::unique_ptr<Eigen::RowVectorXf>(new Eigen::RowVectorXf(topology[i] + 1)));

        //delta and cache initialization
        cacheLayers.push_back(std::unique_ptr<Eigen::RowVectorXf>(new Eigen::RowVectorXf(neuronLayers.size())));
        deltas.push_back(std::unique_ptr<Eigen::RowVectorXf>(new Eigen::RowVectorXf(neuronLayers.size())));

        if(i != topology.size() - 1)
        {
            neuronLayers.back() ->  coeffRef(topology[i]) = 1.0f;
            cacheLayers.back() ->  coeffRef(topology[i]) = 1.0f;
            //.back() for indicating the last added element in the vector
            //coeffRef gives the reference of the value at that place
        }

        //weights matrix initialization
        if(i > 0)
        {
            if(i != topology.size() - 1)
            {
                weights.push_back(std::unique_ptr<Eigen::MatrixXf>(new Eigen::MatrixXf(topology[i - 1] + 1, topology[i] + 1)));
                weights.back() -> setRandom();//TODO what is setRandom?
                weights.back() -> col(topology[i]).setZero();//TODO what is col and setZero
                weights.back() -> coeffRef(topology[i - 1], topology[i]) = 1.0f;
            }
            else
            {
                weights.push_back(std::unique_ptr<Eigen::MatrixXf>(new Eigen::MatrixXf(topology[i - 1] + 1, topology[i])));
                weights.back() -> setRandom();
            }
        }
    }
}

void NeuralNetwork::forwardPropagandation(Eigen::RowVectorXf& input)
{
    //setting the input to input layer
    //                            (startRow, startCol, blockRows, blocCols)
    neuronLayers.front() -> block(0, 0, 1, neuronLayers.front() -> size() - 1) = input;//TODO what is block?

    //propagate data forward and apply activation function to network
    //unaryExpr applies the given function to all elements of CURRENT_LAYER
    for(uint i = 0; i < topology.size(); i++)
    {
        (*neuronLayers[i]) = (*neuronLayers[i - 1]) * (*weights[i - 1]);
        //FIXME
        neuronLayers[i] -> block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));//TODO what is this last part?
    }
}