# NeuralNet
This is a Neural Network developed from scratch using python. The goal is to grasp the main concepts of a neueral network by crafting one!


# The process:

1) Neurons (Nodes):
These are the basic units that the Neural Network is made up of given they are units that recieve input, process the input using what data scientists call "Weights" which are used for training the Neural Network and adjusted for the accuracy of the outcome or the output of the process. The Neural Network is made up of the Input Layer, with one or two hidden layers, and an output layer to match the data gathered as the data is gathered and processed.

2) Weights and Biases:
Every neuron has a "weight" associated with it. The weight(s) are the parameters which are adjusted during the training process of the Neural Network. Biases are additional parameters that allow the network to learn patterns even when all input values are zero.

3) Activation Function:
Neurons apply an activation function to the weighted sum of their inputs, introducing non-linearity to the network. Common activation functions include ReLU (Rectified Linear Unity), sigmoid, and tanh.

4) Layers:

These are the layers that make up the network which Neurons are connected through weights. The input layer recieves the initial data, hidden layers process the data, the output layer produces the final outcome. The deepr the neural network is the more "hidden" layers there are hience the expression deep learning.

5) Architecture: 
The Architecture is the arrangement and connectivity of the layers of neurons. The design of the Architecture accomplishes the different tasks given the implementation and purpose of the Neural network. Connectivity, and the layers of neurons matter! Some of the common architectures are FeedForward Networks, CNNs for Image processing (Convolution Neural Networks), and RNN's (Recurrent Neural Networks) for sequnce data.

6) Loss Function:

Loss function measures how well the networks output matches the expected output. The goal would be to minimize the loss by adjusting the weights and biases during training.

7) Optimization Algorithm:

Like gradient decent, an Optimization Algorithm is used to minimize the loss function by adjusting the weights and biases in the network!

8) Training Data:

Neural Networks learn from examples provided in the training data. The training process involves feeding input data through the network, calculating the loss, and adjusting weights and biases to improve performance.

9) Backpropagation:

 This is the core algorithm for training neural networks. It involves propagating the error backward through the network, updating weights and biases to reduce the error.
