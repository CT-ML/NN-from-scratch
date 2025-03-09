import pandas as pd
import numpy as np

# Define the range for y
y_values = np.linspace(-10, 10, 100)  # 100 points between -10 and 10

# Calculate x = y^2
x_values = y_values**2

# Normalize x and y to [0, 1]
x_normalized = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
y_normalized = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))

# Create a DataFrame
data = pd.DataFrame({
    'x_normalized': x_normalized,
    'y_normalized': y_normalized
})

# Save to CSV
data.to_csv('normalized_data.csv', index=False)