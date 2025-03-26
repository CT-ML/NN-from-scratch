import math
import numpy as np
import csv
import os

def calculate_y(x):



  y = (0.6 * math.sin(math.pi * x) + 
       0.3 * math.sin(3 * math.pi * x) + 
       0.1 * math.sin(5 * math.pi * x) +
       0.05 * math.sin(7 * math.pi * x))

  return y

def generate_training_data(filename, x_min=0, x_max=1, num_points=1000):

    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    
    # Generate evenly spaced x values
    x_values = np.linspace(x_min, x_max, num_points)
    
    # Calculate corresponding y values
    y_values = [calculate_y(x) for x in x_values]
    
    # Write to CSV file
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y'])  # Header row
        for x, y in zip(x_values, y_values):
            writer.writerow([x, y])
    
    print(f"Training data generated and saved to {filename}")
    print(f"Generated {num_points} data points from x={x_min} to x={x_max}")

def main():
    generate_training_data('data/data_test2.csv', -4, 4, 4000)

if __name__ == "__main__":
    main()