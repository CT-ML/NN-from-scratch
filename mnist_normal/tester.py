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

    nn = load_neural_network("trained_nn_on_0.1_error_0.04_learning_rate_0.2_momentum_turn2.pkl")

    y_true = []
    y_pred = []

    for i in range(len(input_test)):
        nn.setInputs(input_test[i])
        nn.forward_calculation()

        probs = nn.nonlinear_output_vector[-1]
        predicted_class = np.argmax(probs)  # Get the indices of the maximum value
        true_class = np.argmax(output_test[i])

        y_true.append(true_class)
        y_pred.append(predicted_class)

        # Console output
        print(f"Sample {i+1}")
        print(f"Predicted Probabilities: {np.round(probs, 3)}")
        print(f"Predicted: {predicted_class}, Actual: {true_class}")
        print("-" * 60)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
