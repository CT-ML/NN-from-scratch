import pickle
import pandas as pd
import os
from main import *
from tabulate import tabulate

def load_neural_network(filename):
    """Load the trained neural network from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def test_neural_network(nn_filename, test_data_filename, threshold=0.1):
    """Load the trained neural network and test it on a given dataset."""
    try:
        # Load the neural network
        nn = load_neural_network(nn_filename)
        
        # Load test data
        test_data = pd.read_csv(test_data_filename)
        
        # Split test data into inputs and expected outputs
        test_inputs = test_data.iloc[:, 1:].values
        test_outputs = test_data.iloc[:, 0].values
        
        # Initialize evaluation metrics
        correct = 0
        false_neg = 0
        false_pos = 0
        
        # Perform evaluation
        for i in range(test_inputs.shape[0]):
            nn.setInputs(test_inputs[i])
            nn.forward_calculation()
            prediction = 1 if nn.nonlinear_output_vector[-1] >= threshold else 0
            
            if prediction == test_outputs[i]:
                correct += 1
            elif test_outputs[i] == 1 and prediction == 0:
                false_neg += 1
            elif test_outputs[i] == 0 and prediction == 1:
                false_pos += 1
        
        # Calculate accuracy
        accuracy = correct / test_inputs.shape[0]
        return (accuracy, correct, false_neg, false_pos)
    
    except Exception as e:
        print(f"Could not evaluate {nn_filename}: {e}")
        return None

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
    """Test all .pkl neural network files in the current directory and print results in a table."""
    test_data_filename = 'data/wisc_bc_test.csv'
    results = []
    
    for filename in os.listdir('.'):
        if filename.endswith('.pkl'):
            nn = load_neural_network(filename)
            learning_rate = nn.learning_rate
            momentum = nn.momentum_turn
            parts = filename.split("error_")

            # Take the part after 'error_' and extract the first number
            error = float(parts[1].split("_")[0])
            result = test_neural_network(filename, test_data_filename)
            if result:
                accuracy, correct, false_neg, false_pos = result
                results.append([learning_rate, momentum, error, f"{accuracy:.4f}", correct, false_neg, false_pos, ])
    
    # Print results as a table
    print(tabulate(results, headers=["Learning Rate", "Momentum", "Error", "Accuracy", "Correct", "False Negatives", "False Positives"], tablefmt="grid"))

if __name__ == "__main__":
    main()
