from main import *

def load_neural_network(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # Load the dataset from the CSV file
    data = pd.read_csv('data/mnist_train_normalized_onehot.csv')

    
    # Set learning parameters
    learning_rate = 0.02
    momentum_turn = 0.2
    error_threshold = 0.066  # Define the error threshold for stopping
    # Create neural network
    nn = load_neural_network("finetuned1_trained_nn_on_0.08_error_0.02_learning_rate_0.2_momentum_turn2.pkl")
    nn.learning_rate = learning_rate
    nn.momentum_turn = momentum_turn
    # Start training loop
    epoch = 0
    while True:
        epoch += 1
        print(f"Epoch {epoch}:")
        shuffled_data = data.sample(frac=1, random_state=None)
        # Split the dataset into inputs and outputs (no shuffle)
        input_data = shuffled_data.iloc[:, 10:].values  # All columns except the first 10 as input
        output_data = shuffled_data.iloc[:, 0:10].values  # First 10 column as output
        print("output shape:"+str(output_data.shape))
        
        total_error = 0
        # Iterate through the dataset for training
        for i in range(input_data.shape[0]):
            # Set inputs and expected output for each training example
            nn.setInputs(input_data[i])
            nn.dataset_outputs = output_data[i]
            # Perform forward calculation
            nn.forward_calculation()
            # Calculate and accumulate the error
            total_error += abs(nn.error_vector)
            nn.backward_calculation()

            # Print the output, desired output, and error for the current input
            # print(f"Sample {i+1} - Predicted Output: {nn.nonlinear_output_vector[-1]} | Desired Output: {nn.dataset_outputs} | Error: {nn.error_vector}")

        avg_error = (total_error / input_data.shape[0]).sum()

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
    save_neural_network(nn, 'finetuned2_trained_nn_on_'+str(error_threshold)+'_error_'+str(learning_rate)+'_learning_rate_'+str(momentum_turn)+'_momentum_turn2.pkl')
    try:
        test_data = pd.read_csv('data/mnist_test_normalized_onehot.csv')
        
        # Split test data
        test_inputs = test_data.iloc[:, 10:].values
        test_outputs = test_data.iloc[:, 0:10].values
       
        
        # Evaluate
        correct = 0
        false_neg = 0
        false_pos = 0
        
        for i in range(test_inputs.shape[0]):
            nn.setInputs(test_inputs[i])
            nn.forward_calculation()
            if nn.nonlinear_output_vector[-1] >= 0.1:
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
 
