import pandas as pd

# Load the dataset
df = pd.read_csv("data/banknote_train.csv")

# Count existing 0s and 1s
num_zeros = df[df.iloc[:, 0] == 0].shape[0]  # First column is the output
num_ones_target = int(num_zeros * (50 / 50))  # Adjusting to reach 65%

# Current number of ones
num_ones_current = df[df.iloc[:, 0] == 1].shape[0]
num_ones_to_add = num_ones_target - num_ones_current

# Duplicate random samples of 1s
ones_to_duplicate = df[df.iloc[:, 0] == 1].sample(n=num_ones_to_add, replace=True)

# Append duplicated rows
df_balanced = pd.concat([df, ones_to_duplicate], ignore_index=True)

# Save the balanced dataset
df_balanced.to_csv("data/banknote_train_balanced_50.csv", index=False)
