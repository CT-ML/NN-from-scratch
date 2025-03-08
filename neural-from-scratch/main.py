import pickle
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


#y is sigmoid(v)
def sigmoid_grad(y):
    return np.multiply(y, 1 - y)


#y is tanh(v)
def tanh_grad(y):
    return np.multiply(1 - np.square(y), 1)


class NeuralNetwork:
    def __init__(self, nb_input, nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn):
        nb_of_neurons_per_layer.insert(0,nb_input)
        self.nb_of_neurons_per_layer =  nb_of_neurons_per_layer
        self.activation_function_array = activation_function_array # activation function at the end of each layer
        self.learning_rate = learning_rate
        self.momentum_turn = momentum_turn
        self.nb_of_layers = len(self.nb_of_neurons_per_layer)

        #TODO optimize vector lengths
        self.internal_activity_vector = [] # output before activation function
        self.nonlinear_output_vector = [] # output after activation function
        self.threshold_vector = []
        self.local_gradient_vector = []
        self.error_vector = []

        for layer in range(self.nb_of_layers): # range from 0 to self.nb_of_layers-1
            currentOut=np.zeros((self.nb_of_neurons_per_layer[layer]))
            currentOut = np.insert(currentOut, 0, -1) if layer!=self.nb_of_layers-1 else currentOut # adding bias to each layer
            self.nonlinear_output_vector.append(currentOut)
            self.internal_activity_vector.append(np.zeros((self.nb_of_neurons_per_layer[layer])))
            self.local_gradient_vector.append(np.zeros((self.nb_of_neurons_per_layer[layer])))
            self.threshold_vector.append(np.zeros(self.nb_of_neurons_per_layer[layer]))

        self.error_vector.append(np.zeros(self.nb_of_neurons_per_layer[self.nb_of_layers - 1]))

        # we are filling all the list[0] with 0 but for activity and other it is meaningless
      

        self.activations = {
            "sigmoid": sigmoid,
            "tanh": tanh
        }
        self.activations_gradient={
            "sigmoid": sigmoid_grad,
            "tanh": tanh_grad
        }
        

        self.dataset_inputs = np.array([])
        self.dataset_outputs = np.array([])
        
        # weight initialization


        # Initialize a list of numpy 2d arrays
        self.weights = []
        self.old_weights = []

        for layer in range(self.nb_of_layers - 1):
            # create neurons_in_layer x neurons_in_next_layer matrix for each layer
            self.weights.append(np.random.randn(len(self.nonlinear_output_vector[layer]),len(self.nonlinear_output_vector[layer + 1])-1 if layer+1 != self.nb_of_layers-1 else len(self.nonlinear_output_vector[layer + 1]))) 
            self.old_weights.append(np.zeros((len(self.nonlinear_output_vector[layer]),len(self.nonlinear_output_vector[layer + 1])-1 if layer+1 != self.nb_of_layers-1 else len(self.nonlinear_output_vector[layer + 1]))))
            #-1 la ma nekhod in consideration l bias tb3 layer l baado
            # l condition statement to take into consideration output layer



            # randomise weights
        # for layer in range(self.nb_of_layers - 1): # layer
        #     for input in range(self.nb_of_neurons_per_layer[layer]): # neuron
        #         for connection in range(self.nb_of_neurons_per_layer[layer + 1]): # connection
        #             self.weights[layer][input, connection] = np.random.rand()


    def forward_calculation(self):
        for layer in range(1, self.nb_of_layers): 
            #TODO add bias
          #  print("self.nonlinear_output_vector["+str(layer-1)+"]: " +str(self.nonlinear_output_vector[layer-1]))
           # print("self.weights["+str(layer-1)+"]: "+str(self.weights[layer-1]))
            self.internal_activity_vector[layer] = np.dot(self.nonlinear_output_vector[layer-1], self.weights[layer-1])
            # apply activation function
            self.nonlinear_output_vector[layer][0 if layer==self.nb_of_layers-1 else 1 :] = self.activations[self.activation_function_array[layer-1]](self.internal_activity_vector[layer])
            #TODO combine
        
        # calculating error
        self.error_vector = self.dataset_outputs - self.nonlinear_output_vector[self.nb_of_layers - 1]

    def setInputs(self, input_data):
        self.nonlinear_output_vector[0][1:] = input_data

    def backward_calculation(self):
        self.local_gradient_vector[self.nb_of_layers - 1] = self.error_vector * \
            (self.activations_gradient[self.activation_function_array[self.nb_of_layers - 2]](self.nonlinear_output_vector[self.nb_of_layers - 1]))


        for layer in range(self.nb_of_layers-2, 0, -1):

            self.local_gradient_vector[layer] = (self.activations_gradient[self.activation_function_array[layer]](self.nonlinear_output_vector[layer][1:])) * \
                np.dot(self.weights[layer][1:], self.local_gradient_vector[layer + 1])

        for layer in range( self.nb_of_layers - 1):
            delta_weights = self.momentum_turn * (self.weights[layer]-self.old_weights[layer]) \
                + self.learning_rate * np.outer(self.nonlinear_output_vector[layer], self.local_gradient_vector[layer + 1])
            self.old_weights[layer] = self.weights[layer]

            self.weights[layer] = self.weights[layer] + delta_weights
            
        print("current weights: "+str(self.weights))
        print("old weights: "+str(self.old_weights))

            
def main():
    # Define network parameters
    nb_of_neurons_per_layer = [3, 2, 1]  # Input layer, one hidden layer, output layer
    activation_function_array = ["sigmoid", "sigmoid", "sigmoid"]  # Activation functions per layer
    learning_rate = 0.01
    momentum_turn = 0.7
    nb_input=3

    # Create neural network
    nn = NeuralNetwork(nb_input, nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn)

    # Dummy input
    input_data = np.array([1, 1, 1])

    # Assign input to the first layer's nonlinear output vector
    nn.setInputs(input_data)

    nn.dataset_outputs = np.array([0])

    # Perform forward calculation
    nn.forward_calculation()

    # Print the output of the last layer
    print("Output:", nn.nonlinear_output_vector[-1])
    print("Error:", nn.error_vector)
    nn.backward_calculation()
    nn.forward_calculation()
    print("Output:", nn.nonlinear_output_vector[-1])
    print("Error:", nn.error_vector)

if __name__ == "__main__":
    main()
 