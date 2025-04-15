import pickle
import pandas as pd
import numpy as np
from main import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def load_neural_network(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    test_data = pd.read_csv("data/mnist_test_normalized_onehot.csv")
    input_test = test_data.iloc[:, 10:].values  
    output_test = test_data.iloc[:, 0:10].values  # One-hot encoded labels

    nn = load_neural_network("20 20 15/finetuned2_trained_nn_on_0.066_error_0.02_learning_rate_0.2_momentum_turn2.pkl")

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
