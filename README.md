# ProjectVale
First project of an AI neural network made by me.

## Introduction to Neural Network
[source](https://becominghuman.ai/artificial-neuron-networks-basics-introduction-to-neural-networks-3082f1dcca8c)  
A Neural Network is a computational model based on structure and functions of biological neural networks.  
There are three different layers:

- **Input Layer**  
 Reads all the inputs which are transferred to hidden layers.
 Communicates with the external enviroment that gives information to the neural network. 
 It represents the condition to train the neural network.
- **Hidden Layer**  
 Is a collection of neurons which has activation function applied on and their scope is to process inputs received from input layers: it extracts the required features from data.
 They can be more than one.
- **Output Layer**  
 Collects and transmits the data processed from hidden layers in the way it has been designed. 
 They can be traced back to input layers.


## Code description
### NeuralNetwork class
- **Constructor**  
 Topology vector specifies how many neurons are in each layer and its size is equal to a number of layers in the neural network. Layers in the neural network are arrays of neurons and they are stored as vector. Each element has the activation value of neuron in that layer. All neurons remains "activated" except the ones in the output layer
- **forwardPropagandation**  
 The neuron of the current layer gets it's input by a dot product between neuronLayers vector and the other neurons of the layer: it takes inputs multiplied by weights and also adds the bias term. The last column of weights matrix is initialised to zero except for the last element, which is set to 1. By doing so, the bias neuron of the current layer, takes input of the bias neuron of the previous layer only.