import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def normalize_csv(file_path, output_path):
    # Load CSV file
    df = pd.read_csv(file_path)
    
    # Drop the first column
    df = df.iloc[:, 1:]
    
    # Replace 'B' with 0 and 'M' with 1 in the first column
    df.iloc[:, 0] = df.iloc[:, 0].replace({'B': 0, 'M': 1})

    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Apply Min-Max normalization to each numerical column
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())

    # Save the normalized data
    df.to_csv(output_path, index=False)
    print(f"Normalized CSV saved to {output_path}")

# Example usage
# input_file = "data/wisc_bc_data.csv"
# output_file = "data/wisc_bc_data_normalized.csv"
# normalize_csv(input_file, output_file)


    # # Example usage
# input_file = "data/Banknote_Excel.csv"
# output_file = "data/banknote_normalized.csv"
# normalize_csv(input_file, output_file)

def split_csv(file_path, train_output, test_output, train_ratio=0.8):
    # Load CSV file
    df = pd.read_csv(file_path)

    # Shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # Compute split index
    split_index = int(len(df) * train_ratio)

    # Split into training and testing sets
    train_df = df[:split_index]
    test_df = df[split_index:]

    # Save to new CSV files
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"Training set saved to {train_output} ({len(train_df)} rows)")
    print(f"Testing set saved to {test_output} ({len(test_df)} rows)")

# # Example usage
input_file = "data/banknote_normalized.csv"
train_file = "data/banknote_train.csv"
test_file = "data/banknote_test.csv"
split_csv(input_file, train_file, test_file)



