import pandas as pd
import numpy as np
import pickle
from main import *

def load_neural_network(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # Load the dataset from the CSV file
    data = pd.read_csv('data/banknote_train_balanced.csv')

    # Load the pre-trained neural network
    pkl_filename = "finetuned_dup_32,32/finetuned3_nn_on_0.012_error_0.02_learning_rate_0.2_momentum_turn2.pkl"
    nn = load_neural_network(pkl_filename)
    print("Loaded neural network from", pkl_filename)

    # Set learning parameter
    learning_rate = 0.01
    momentum_turn = 0.1
    error_threshold = 0.01  # Define the error threshold for stopping

    nn.learning_rate = learning_rate
    nn.momentum_turn = momentum_turn

    # Start training loop (fine-tuning)
    epoch = 0
    while True:
        epoch += 1
        # print(f"Epoch {epoch}:")
        shuffled_data = data.sample(frac=1, random_state=None)
        input_data = shuffled_data.iloc[:, 1:].values  # All columns except the first as input
        output_data = shuffled_data.iloc[:, 0].values  # First column as output

        total_error = 0
        for i in range(input_data.shape[0]):
            nn.setInputs(input_data[i])
            nn.dataset_outputs = np.array([output_data[i]])

            nn.forward_calculation()
            total_error += abs(nn.error_vector)
            nn.backward_calculation()

        avg_error = total_error / input_data.shape[0]
        print(f"Average Error for Epoch {epoch}: {avg_error}")

        if avg_error < error_threshold:
            print("Fine-tuning complete. Error is below threshold.")
            break

    print("Final output:", nn.nonlinear_output_vector[-1])
    print("Final error:", nn.error_vector)
    print("Now for testing data")

    fine_tuned_filename = 'finetuned4_nn_on_'+str(error_threshold)+'_error_'+str(learning_rate)+'_learning_rate_'+str(momentum_turn)+'_momentum_turn2.pkl'

    # Save the fine-tuned model
    save_neural_network(nn, fine_tuned_filename)
    print(f"Fine-tuned model saved as {fine_tuned_filename}")

    # Testing
    try:
        test_data = pd.read_csv('data/banknote_test.csv')
        test_inputs = test_data.iloc[:, 1:].values
        test_outputs = test_data.iloc[:, 0].values
        
        correct = 0
        false_neg = 0
        false_pos = 0
        
        for i in range(test_inputs.shape[0]):
            nn.setInputs(test_inputs[i])
            nn.forward_calculation()
            prediction = 1 if nn.nonlinear_output_vector[-1] >= 0.1 else 0
            
            if prediction == test_outputs[i]:
                correct += 1
            elif test_outputs[i] == 1 and prediction == 0:
                false_neg += 1
                print("for false_neg", nn.nonlinear_output_vector[-1])
            elif test_outputs[i] == 0 and prediction == 1:
                false_pos += 1
        
        print(f"\nTest Results:")
        print(f"Accuracy: {correct/test_inputs.shape[0]:.4f}")
        print("Correct:", correct)
        print(f"False Negatives: {false_neg}")
        print(f"False Positives: {false_pos}")
        
    except Exception as e:
        print(f"Could not evaluate test data: {e}")

if __name__ == "__main__":
    main()
