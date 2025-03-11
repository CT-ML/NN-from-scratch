import pickle
import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def step_function(x):
    return np.where(x >= 0, 1, 0)

def step_function_grad(y):
    return np.zeros_like(y)


#y is sigmoid(v)
def sigmoid_grad(y):
    return np.multiply(y, 1 - y)


#y is tanh(v)
def tanh_grad(y):
    return np.multiply(1 - np.square(y), 1)


class NeuralNetwork:
    def __init__(self, nb_input, nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn):
        # inserting first layer (inputs) from main no need for this
        # nb_of_neurons_per_layer.insert(0, nb_input)
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
            "tanh": tanh,
            "step": step_function,
        }
        self.activations_gradient={
            "sigmoid": sigmoid_grad,
            "tanh": tanh_grad,
            "step": step_function_grad,
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
            
        # print("current weights: "+str(self.weights))
        # print("old weights: "+str(self.old_weights))
def duplicate_positive(n):
    # Load the dataset
    data = pd.read_csv('data/wisc_bc_train.csv')

    # Find rows where first column equals 1 (cancer cases)
    cancer_rows = data[data.iloc[:, 0] == 1]
    
    # Start with the original dataset
    new_data = data.copy()
    
    # Add the cancer rows n times
    for _ in range(n):
        new_data = pd.concat([new_data, cancer_rows], ignore_index=True)

    # Save the new dataset
    new_data.to_csv('data/wisc_bc_train_duplicated.csv', index=False)


def main():
    # Load the dataset from the CSV file
    duplicate_positive(2)
    data = pd.read_csv('data/wisc_bc_train_duplicated.csv')

    # Define the number of neurons per layer (dynamic based on input size)
    input_size = data.shape[1] - 1  # Assuming the first column is the output
    nb_of_neurons_per_layer = np.array([input_size, 32, 16, 1])  # Array with decreasing neurons for each layer

    # Define activation functions per layer (sigmoid for all layers)
    activation_function_array = []
    for _ in range(len(nb_of_neurons_per_layer) - 1):
        activation_function_array.append("sigmoid")
    

    print(activation_function_array)

    # Set learning parameters
    learning_rate = 0.05
    momentum_turn = 0.3
    error_threshold = 0.02  # Define the error threshold for stopping

    # Create neural network
    nn = NeuralNetwork(input_size, nb_of_neurons_per_layer, activation_function_array, learning_rate, momentum_turn)
    print(nb_of_neurons_per_layer)
    # Start training loop
    epoch = 0
    while True:
        epoch += 1
        print(f"Epoch {epoch}:")
        shuffled_data = data.sample(frac=1, random_state=None)
        # Split the dataset into inputs and outputs (no shuffle)
        input_data = shuffled_data.iloc[:, 1:].values  # All columns except the first as input
        output_data = shuffled_data.iloc[:, 0].values  # First column as output

        total_error = 0
        # Iterate through the dataset for training
        for i in range(input_data.shape[0]):
            # Set inputs and expected output for each training example
            nn.setInputs(input_data[i])
            nn.dataset_outputs = np.array([output_data[i]])

            # Perform forward calculation
            nn.forward_calculation()

            # Calculate and accumulate the error
            total_error += abs(nn.error_vector)
            nn.backward_calculation()

            # Print the output, desired output, and error for the current input
            print(f"Sample {i+1} - Predicted Output: {nn.nonlinear_output_vector[-1]} | Desired Output: {nn.dataset_outputs} | Error: {nn.error_vector}")

        avg_error = total_error / input_data.shape[0]

        # Average error for the epoch
        print(f"Average Error for Epoch {epoch}: {avg_error}")

        # Stop if error is below the threshold
        if avg_error < error_threshold:
            print("Training complete. Error is below threshold.")
            break

    # Final output and error after training
    print("Final output:", nn.nonlinear_output_vector[-1])
    print("Final error:", nn.error_vector)
    print("now for testing data")
    try:
        test_data = pd.read_csv('data/wisc_bc_test.csv')
        
        # Split test data
        test_inputs = test_data.iloc[:, 1:].values
        test_outputs = test_data.iloc[:, 0].values
        
        # Evaluate
        correct = 0
        false_neg = 0
        false_pos = 0
        
        for i in range(test_inputs.shape[0]):
            nn.setInputs(test_inputs[i])
            nn.forward_calculation()
            if nn.nonlinear_output_vector[-1] >= 0.2:
                prediction = 1
            else:
                prediction = 0
            
            if prediction == test_outputs[i]:
                correct += 1
            elif test_outputs[i] == 1 and prediction == 0:
                false_neg += 1
                print("for false_neg" + str(nn.nonlinear_output_vector[-1]))
            elif test_outputs[i] == 0 and prediction == 1:
                false_pos += 1
        
        print(f"\nTest Results:")
        print(f"Accuracy: {correct/test_inputs.shape[0]:.4f}")
        print("Correct: " + str(correct))
        print(f"False Negatives: {false_neg}")
        print(f"False Positives: {false_pos}")
        
    except Exception as e:
        print(f"Could not evaluate test data: {e}")

if __name__ == "__main__":
    main()
 
