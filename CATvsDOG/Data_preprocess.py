import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
np.random.seed(42)

# Define paths
BASE_DIR = 'PetImages'
CAT_DIR = os.path.join(BASE_DIR, 'Cat')
DOG_DIR = os.path.join(BASE_DIR, 'Dog')

# Create output directories if needed
def create_directories():
    os.makedirs('processed_data', exist_ok=True)
    print("Created directory: processed_data")

def process_dataset():
    print("Starting dataset preparation...")
    
    # Create directory structure
    create_directories()
    
    # Get all valid image paths
    cat_images = []
    dog_images = []
    
    # Process cat images
    print("Processing cat images...")
    for img_name in os.listdir(CAT_DIR):
        img_path = os.path.join(CAT_DIR, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    # Check if image is RGB
                    if img.mode == 'RGB':
                        cat_images.append(img_path)
            except Exception as e:
                print(f"Skipping corrupted image {img_path}: {e}")
    
    # Process dog images
    print("Processing dog images...")
    for img_name in os.listdir(DOG_DIR):
        img_path = os.path.join(DOG_DIR, img_name)
        if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    # Check if image is RGB
                    if img.mode == 'RGB':
                        dog_images.append(img_path)
            except Exception as e:
                print(f"Skipping corrupted image {img_path}: {e}")
    
    # Split data
    cat_train, cat_test = train_test_split(cat_images, test_size=0.1, random_state=42)
    dog_train, dog_test = train_test_split(dog_images, test_size=0.1, random_state=42)
    
    print(f"Cat images - Train: {len(cat_train)}, Test: {len(cat_test)}")
    print(f"Dog images - Train: {len(dog_train)}, Test: {len(dog_test)}")
    
    # Create arrays to store our data
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    # Process training cats (label 0)
    print("Processing training cat images...")
    for img_path in cat_train:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (100, 100))  # Resize to 100x100
            train_images.append(img)
            train_labels.append(0)  # 0 for cat
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process training dogs (label 1)
    print("Processing training dog images...")
    for img_path in dog_train:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (100, 100))  # Resize to 100x100
            train_images.append(img)
            train_labels.append(1)  # 1 for dog
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process test cats (label 0)
    print("Processing test cat images...")
    for img_path in cat_test:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (100, 100))  # Resize to 100x100
            test_images.append(img)
            test_labels.append(0)  # 0 for cat
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Process test dogs (label 1)
    print("Processing test dog images...")
    for img_path in dog_test:
        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = cv2.resize(img, (100, 100))  # Resize to 100x100
            test_images.append(img)
            test_labels.append(1)  # 1 for dog
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert to numpy arrays
    X_train = np.array(train_images, dtype='float32') / 255.0  # Normalize to [0,1]
    y_train = np.array(train_labels)
    X_test = np.array(test_images, dtype='float32') / 255.0  # Normalize to [0,1]
    y_test = np.array(test_labels)
    
    print(f"Dataset prepared successfully!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Save the processed dataset
    print("Saving processed dataset...")
    np.save('processed_data/X_train.npy', X_train)
    np.save('processed_data/y_train.npy', y_train)
    np.save('processed_data/X_test.npy', X_test)
    np.save('processed_data/y_test.npy', y_test)
    
    print("Dataset saved! You can load it later using np.load()")
    
    # Return the data for immediate use if needed
    return X_train, y_train, X_test, y_test



# Run the processing
if __name__ == "__main__":
    process_dataset()
    