import pickle
import pandas as pd
import os
from main import *
from tabulate import tabulate
import matplotlib.pyplot as plt


    

def load_neural_network(filename):
    """Load the trained neural network from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def extract_params(filename):
    """Extract error, learning rate, and momentum from the filename without regex."""
    parts = filename.replace(".pkl", "").split("_")
    try:
        error_index = parts.index("error") + 1
        learning_rate_index = parts.index("learning") + 2  # 'learning_rate'
        momentum_index = parts.index("momentum") + 1
        error = float(parts[error_index])
        learning_rate = float(parts[learning_rate_index])
        momentum = float(parts[momentum_index])
        return error, learning_rate, momentum
    except (ValueError, IndexError):
        return None, None, None

def main():
    test_data = pd.read_csv("data/data_test2.csv")
    input_test = test_data.iloc[:, 0].values  # Assuming first column is input
    output_test = test_data.iloc[:, 1].values  # Assuming second column is expected output

    # Initialize neural network
    nn = load_neural_network("-1 to 1 0.015 e 0.02 lr 0.1m.pkl")  # Ensure this matches your class constructor

    # Run the test dataset through the trained network
    predicted_test_values = []
    for i in range(input_test.shape[0]):
        nn.setInputs(input_test[i])
        nn.forward_calculation()
        predicted_test_values.append(nn.nonlinear_output_vector[-1].copy())

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.scatter(input_test, output_test, color='blue', label='Expected')
    plt.scatter(input_test, predicted_test_values, color='red', label='Predicted')
    plt.title("Neural Network Test Results")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
