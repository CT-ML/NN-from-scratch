import pickle
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def step_function(x):
    return np.where(x >= 0, 1, 0)

class NeuralNetwork:
    def __init__(self, nb_input, nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn):
        self.nb_of_neurons_per_layer = [nb_input] + nb_of_neurons_per_layer
        self.activation_function_array = activation_function_array # activation function at the end of each layer
        self.learning_rate = learning_rate
        self.momentum_turn = momentum_turn
        self.nb_of_layers = len(self.nb_of_neurons_per_layer)

        #TODO optimize vector lengths
        self.internal_activity_vector = [] # output before activation function
        self.nonlinear_output_vector = [] # output after activation function
        self.threshold_vector = []
        self.local_gradient_vector = []

        self.nonlinear_output_vector.append(np.zeros((self.nb_of_neurons_per_layer[0]))) # treat input vector as imaginary layer
        for layer in range(1, self.nb_of_layers-1, 1):
            self.internal_activity_vector.append(np.zeros((self.nb_of_neurons_per_layer[layer])))
            self.nonlinear_output_vector.append(np.zeros((self.nb_of_neurons_per_layer[layer])))
            self.local_gradient_vector.append(np.zeros((self.nb_of_neurons_per_layer[layer])))
            self.threshold_vector.append(np.zeros(self.nb_of_neurons_per_layer[layer]))

        

        self.activations = {
            "sigmoid": sigmoid,
            "tanh": tanh,
            "step": step_function
        }

        self.dataset_inputs = np.array([])
        self.dataset_outputs = np.array([])
        
        # weight initialization


        # Initialize a lsit of numpy 2d arrays
        self.weights = []

        for layer in range(self.nb_of_layers - 2):
            # create neurons_in_layer x neurons_in_next_layer matrix for each layer
            self.weights.append(np.random.randn(self.nb_of_neurons_per_layer[layer],self.nb_of_neurons_per_layer[layer + 1]))


            # randomise weights
        # for layer in range(self.nb_of_layers - 1): # layer
        #     for input in range(self.nb_of_neurons_per_layer[layer]): # neuron
        #         for connection in range(self.nb_of_neurons_per_layer[layer + 1]): # connection
        #             self.weights[layer][input, connection] = np.random.rand()


    def forward_calculation(self):
        for layer in range(1, self.nb_of_layers - 1):
            #TODO add bias
            self.internal_activity_vector[layer] = np.dot(self.nonlinear_output_vector[layer-1], self.weights[layer-1])
            # apply activation function
            self.nonlinear_output_vector[layer] = self.activations[self.activation_function_array[layer-1]](self.internal_activity_vector[layer])
            #TODO combine
def main():
    # Define network parameters
    nb_of_neurons_per_layer = [1, 1, 1]  # Input layer, one hidden layer, output layer
    activation_function_array = ["sigmoid", "sigmoid", "sigmoid"]  # Activation functions per layer
    learning_rate = 0.01
    momentum_turn = 0.9

    # Create neural network
    nn = NeuralNetwork(nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn)

    # Dummy input
    input_data = np.array([1])

    # Assign input to the first layer's nonlinear output vector
    nn.nonlinear_output_vector[0] = input_data

    # Perform forward calculation
    nn.forward_calculation()

    # Print the output of the last layer
    print("Output:", nn.nonlinear_output_vector[-1])

if __name__ == "__main__":
    main()
