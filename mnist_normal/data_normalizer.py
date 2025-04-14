#!/usr/bin/env python3
import numpy as np

def one_hot_encode(label, num_classes=10):
    """Convert a digit (0-9) to one-hot encoded vector"""
    one_hot = np.zeros(num_classes)
    one_hot[int(label)] = 1
    return one_hot

def normalize_mnist_with_onehot(in_path: str, out_path: str) -> None:
    # Load entire dataset (labels + pixels)
    # Assumes no header, comma-delimited, shape = (n_samples, 785)
    data = np.loadtxt(in_path, delimiter=',')
    
    # Extract labels (first column) and features (remaining columns)
    labels = data[:, 0]
    features = data[:, 1:]
    
    # Normalize pixel values to [0,1]
    features /= 255.0
    
    # One-hot encode the labels
    onehot_labels = np.array([one_hot_encode(label) for label in labels])
    
    # Create a new dataset with one-hot encoded labels and normalized features
    # Shape will be (n_samples, 10+784) = (n_samples, 794)
    combined_data = np.hstack((onehot_labels, features))
    
    # Save to CSV
    np.savetxt(out_path, combined_data, delimiter=',', fmt='%.6g')
    print(f"Processed {len(labels)} samples: normalized features and one-hot encoded labels")
    print(f"New data shape: {combined_data.shape} (10 one-hot columns + 784 feature columns)")

def main():
    output="data/mnist_train_normalized_onehot.csv"
    input="data/mnist_train.csv"
    normalize_mnist_with_onehot(input, output)
    print(f"Done: normalized data with one-hot labels written to {output}")
    
    output="data/mnist_test_normalized_onehot.csv"
    input="data/mnist_test.csv"
    normalize_mnist_with_onehot(input, output)
    print(f"Done: normalized data with one-hot labels written to {output}")

if __name__ == "__main__":
    main()