import pickle
import numpy as np
from main import *
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

# Load the trained neural network
def load_neural_network(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Canvas application for drawing
class DigitDrawer:
    def __init__(self, nn):
        self.nn = nn
        self.canvas_size = 280  # 280x280 pixels (10x scale of 28x28)
        self.brush_size = 7

        self.root = tk.Tk()
        self.root.title("Draw a digit (0-9)")

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.label_result = tk.Label(self.root, text="Prediction: None", font=("Helvetica", 18))
        self.label_result.pack()

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # White background
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict_digit)

        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)  # Draw on PIL image (0 = black)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=255)
        self.label_result.config(text="Prediction: None")

    def preprocess(self):
        # Resize image to 28x28 and invert (white bg to black bg)
        img_resized = self.image.resize((28, 28), Image.LANCZOS)
        img_inverted = ImageOps.invert(img_resized)

        # Normalize pixels to range [0, 1]
        img_array = np.asarray(img_inverted) / 255.0
        img_flat = img_array.flatten()

        return img_flat

    def predict_digit(self, event=None):
        input_data = self.preprocess()

        self.nn.setInputs(input_data)
        self.nn.forward_calculation()
        probs = self.nn.nonlinear_output_vector[-1]
        predicted_class = np.argmax(probs)

        self.label_result.config(text=f"Prediction: {predicted_class} \n(Probs: {np.round(probs, 2)})")
        print(f"Predicted Probabilities: {np.round(probs, 3)}")
        print(f"Predicted: {predicted_class}")
        print("-" * 60)

# Entry point
def main():
    nn = load_neural_network("20 20 20/finetuned2_trained_nn_on_0.04_error_0.01_learning_rate_0.2_momentum_turn2.pkl")
    DigitDrawer(nn)

if __name__ == "__main__":
    main()
