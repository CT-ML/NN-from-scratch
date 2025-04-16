from main import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_neural_network(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    # Load the dataset from the CSV file
    data = pd.read_csv('data/mnist_train_normalized_onehot.csv')

    
    # Set learning parameters
    learning_rate = 0.01
    momentum_turn = 0.2
    error_threshold = 0.03  # Define the error threshold for stopping
    # Create neural network
    nn = load_neural_network("25 25 25/finetuned3_trained_nn_on_0.04_error_0.01_learning_rate_0.2_momentum_turn2.pkl")
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
    save_neural_network(nn, 'finetuned4_trained_nn_on_'+str(error_threshold)+'_error_'+str(learning_rate)+'_learning_rate_'+str(momentum_turn)+'_momentum_turn2.pkl')
    test_data = pd.read_csv("data/mnist_test_normalized_onehot.csv")
    input_test = test_data.iloc[:, 10:].values  
    output_test = test_data.iloc[:, 0:10].values  # One-hot encoded labels

    nn = load_neural_network("25 25 25/trained_nn_on_0.1_error_0.04_learning_rate_0.2_momentum_turn2.pkl")

    y_true = []
    y_pred = []

    # List to store incorrect predictions (images, true labels, predicted labels)
    incorrect_predictions = []

    for i in range(len(input_test)):
        nn.setInputs(input_test[i])
        nn.forward_calculation()

        probs = nn.nonlinear_output_vector[-1]
        predicted_class = np.argmax(probs)  # Get the index of the maximum value
        true_class = np.argmax(output_test[i])

        y_true.append(true_class)
        y_pred.append(predicted_class)

        # If prediction is incorrect, save the image and label information
        if predicted_class != true_class:
            incorrect_predictions.append((input_test[i], true_class, predicted_class, probs))

        # Console output
        print(f"Sample {i+1}")
        print(f"Predicted Probabilities: {np.round(probs, 3)}")
        print(f"Predicted: {predicted_class}, Actual: {true_class}")
        print("-" * 60)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    incorrect_count = np.sum(cm) - np.trace(cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix\nIncorrect Predictions: {incorrect_count}")
    plt.show()

    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    # Plot incorrect predictions
    if incorrect_predictions:
        print(f"\nTotal incorrect predictions: {len(incorrect_predictions)}")
        
        # Display the incorrect predictions
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))
        axes = axes.flatten()

        for i, (image, true_class, pred_class, probs) in enumerate(incorrect_predictions[:9]):  # Display first 9 incorrect predictions
            ax = axes[i]
            image = image.reshape(28, 28)  # Reshape the flat vector to 28x28 for MNIST images
            ax.imshow(image, cmap='gray')
            ax.set_title(f"True: {true_class}, Pred: {pred_class}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

 
