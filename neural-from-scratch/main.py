import pickle
import numpy as np


class NeuralNetwork:
    def __init__(self, nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn):
        self.nb_of_neurons_per_layer = nb_of_neurons_per_layer
        self.activation_function_array = activation_function_array # activation function at the end of each layer
        self.learning_rate = learning_rate
        self.momentum_turn = momentum_turn
        self.nb_of_layers = len(nb_of_neurons_per_layer)

        #TODO optimize vector lengths
        self.internal_activity_vector = np.zeros(self.nb_of_layers - 1, self.nb_of_neurons_per_layer[1]) # output before activation function
        self.nonlinear_output_vcetor = np.zeros(self.nb_of_layers - 1, self.nb_of_neurons_per_layer[1])

        self.local_gradient_vector = np.zeros(self.nb_of_layers - 1, self.nb_of_neurons_per_layer[1])

        self.threshold_vector = np.zeros(self.nb_of_layers - 1)

        # TODO define activation functions: sigmoide, tanh, ...

        self.dataset_inputs = np.array([])
        self.dataset_outputs = np.array([])
        
        # weight initialization


        # Initialize a 3D list with all elements set to 0
        self.weights = []

        for layer in range(len(nb_of_neurons_per_layer) - 1):
            layer_weights = []
            
            # Number of neurons in the current layer
            neurons_in_layer = nb_of_neurons_per_layer[layer]
            
            # Number of neurons in the next layer
            neurons_in_next_layer = nb_of_neurons_per_layer[layer + 1]
            
            for neuron in range(neurons_in_layer):
                # Create a list of 0's for the current neuron, with size equal to the number of neurons in the next layer
                neuron_weights = [0 for _ in range(neurons_in_next_layer)]
                layer_weights.append(neuron_weights)
            
            self.weights.append(layer_weights)

            # randomise weights
            for layer in range(self.nb_of_layers - 1): # layer
                for input in range(self.nb_of_neurons_per_layer[layer]): # neuron
                    for connection in range(self.nb_of_neurons_per_layer[layer + 1]): # connection
                        self.weights[layer][input][connection] = np.random.rand()


    def forward_calculation(self):
        for layer in range(self.nb_of_layers):
            # for each neuron in next layer, iterate through neurons in previous layer
            for next_neuron in range(self.nb_of_neurons_per_layer[layer + 1]):
                self.internal_activity_vector[layer + 1][next_neuron] = threshold_vector[layer] * -1
                for previous_neuron in range(self.nb_of_neurons_per_layer[layer]):
                    self.internal_activity_vector[layer + 1][next_neuron] += weights[layer][previous_neuron][next_neuron] * self.nonlinear_output_vector[layer][previous_neuron]
            # apply activation funcion
            self.nonlinear_output_vector[layer][next_neuron] = self.activation_function(self.internal_activity_vector[layer][previous_neuron])
