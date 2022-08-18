#include "NeuralNetwork.h"

//activation function
constexpr float activationFunction(float x)
{
    return tanhf(x);//TODO what is tanhf?
}

constexpr float activationFunctionDerivative(float x)
{
    return 1 - tanhf(x) * tanhf(x);
}

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
        neuronLayers[i] -> block(0, 0, 1, topology[i]).unaryExpr(std::ptr_fun(activationFunction));//TODO what is this last part? 
    }
}

void NeuralNetwork::backwardPropagandation(Eigen::RowVectorXf& output)
{
    calcError(output);
    updateWeights();
}

void NeuralNetwork::calcError(Eigen::RowVectorXf& output)
{
    //determine errors made by neurons of last layer
    (*deltas.back()) = output - (*neuronLayers.back());//deference because they are pointers

    //error calculation of hidden layers
    //we start from the last hidden layer
    //and continue till the first hidden layer
    for (uint i = topology.size() - 2; i > 0; i--)
    {
        (*deltas[i]) = (*deltas[i + 1]) * (weights[i] -> transpose());//TODO what is transpose?
    }
}

void NeuralNetwork::updateWeights()
{
    for (uint i = 0; i < topology.size() - 1; i++)
    {
        //iterating over the different layers (from first hidden to output layer)
        //if layer is the output, there no bias neuron, number of neurons specified = number of cols
        //if layer is not output, thiere is bias neuron and number of neurons specified = number of cols - 1
        if (i != topology.size() - 2)
        {
            for (uint c = 0; c < weights[i] -> cols() - 1; c++)
            {
                for (uint r = 0; r < weights[i] -> rows(); r++)
                {
                    weights[i] -> coeffRef(r, c) += learningRate * deltas[i + 1] -> coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1] -> coeffRef(c) * neuronLayers[i] -> coeffRef(r));
                }
            }
        }
        else
        {
            for (uint c = 0; c < weights[i] -> cols(); c++)
            {
                for(uint r = 0; r < weights[i] -> rows(); r++)
                {
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::train(std::vector<std::unique_ptr<Eigen::RowVectorXf>>& input_data, std::vector<std::unique_ptr<Eigen::RowVectorXf>>& output_data)
{
    for(uint i = 0; i < input_data.size(); i++)
    {
        std::cout << "Input to neural network is: " << *input_data[i] << std::endl;
        forwardPropagandation(*input_data[i]);
        std::cout << "Expected output is: " << *output_data[i] << std::endl;
        std::cout << "Output produced is: " << *neuronLayers.back() << std::endl;
        backwardPropagandation(*output_data[i]);
        std::cout << "MSE: " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back() -> size()) << std::endl; //TODO what is MSE?
    }
}