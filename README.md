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

===========================================================================================================
These are a list of Neural Network Architectures I recieved after asking ChatGPT about the different models.

    Feedforward Neural Networks (FNN):
        Basic neural network architecture where information travels in one direction, from the input layer to the output layer.
        No cycles or loops in the network structure.

    Convolutional Neural Networks (CNN):
        Particularly effective for tasks involving images and spatial data.
        Uses convolutional layers to detect local patterns and spatial hierarchies.
        Often includes pooling layers for down-sampling and reducing the dimensionality of the data.

    Recurrent Neural Networks (RNN):
        Suitable for sequence data, such as time series or natural language.
        Utilizes feedback connections to allow information persistence over time.
        Well-suited for tasks involving dependencies on previous inputs.

    Long Short-Term Memory Networks (LSTM):
        A type of recurrent neural network designed to address the vanishing gradient problem.
        Effective for learning long-term dependencies in sequential data.

    Gated Recurrent Unit (GRU):
        Similar to LSTM but with a simplified structure, making it computationally more efficient.
        Used for similar tasks as LSTM, such as sequence modeling.

    Autoencoders:
        Consists of an encoder and a decoder, designed to learn efficient representations of data.
        Used for tasks like data compression, denoising, and dimensionality reduction.

    Generative Adversarial Networks (GAN):
        Comprises a generator and a discriminator that are trained simultaneously through adversarial training.
        Used for generating new data instances, often in the form of realistic images.

    Radial Basis Function Networks (RBFN):
        Employs radial basis functions as activation functions in the hidden layer.
        Commonly used for pattern recognition and function approximation.

    Modular Neural Networks:
        Composed of multiple networks that specialize in different subtasks.
        Enables the development of more modular and interpretable models.

    Capsule Networks (CapsNets):
        Proposed as an alternative to traditional convolutional neural networks, aiming to address issues related to viewpoint variations and part-whole relationships.

    Transformers:
        Originally designed for natural language processing tasks.
        Utilizes attention mechanisms to focus on different parts of the input sequence simultaneously.
        Widely adopted in various tasks beyond NLP, such as computer vision.

