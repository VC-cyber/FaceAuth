import cv2
import os
import numpy as np
from model import getRecognitionModel
import random

# Directory where preprocessed images are saved
preprocessed_dir = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/preprocessed_data'

# Initialize the face recognizer
face_recognizer = getRecognitionModel()

# Adjust the chunk size as needed
chunk_size = 100  # Define the chunk size for training

# Subset size (0 to use all images)
# subset_size = 100

# Resize dimensions (smaller images = smaller model)
resize_dim = (200, 200)

model_save_path = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Weights/face_recognizer_model.yml'

if os.path.exists(model_save_path):
        os.remove(model_save_path)
        print(f"File '{model_save_path}' has been deleted.")
else:
    print(f"File '{model_save_path}' does not exist.")


def get_subset(data, subset_size):
    """
    Returns a subset of the data while ensuring the last three data points are included.
    
    Parameters:
    data (list): The original dataset.
    subset_size (int): The size of the desired subset (including the last three data points).

    Returns:
    list: The subset of the data.
    """
    # Ensure subset_size is at least 3
    if subset_size < 3:
        raise ValueError("Subset size must be at least 3 to include the last three data points.")
    
    # The number of random samples to select
    num_random_samples = subset_size - 3
    
    # Select random samples from the dataset excluding the last three data points
    random_samples = random.sample(data[:-3], num_random_samples)
    
    # Add the last three data points to the subset
    subset = random_samples + data[-3:]
    
    return subset

def load_chunk(chunk_start, chunk_end):
    faces = []
    labels = []

    with open(os.path.join(preprocessed_dir, 'labels.txt'), 'r') as label_file:
        lines = label_file.readlines()[chunk_start:chunk_end]

        for line in lines:
            face_filename, label = line.strip().split(',')
            face_path = os.path.join(preprocessed_dir, face_filename)
            face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
            
            # Resize image to a smaller size
            face = cv2.resize(face, resize_dim)
            
            faces.append(face)
            labels.append(int(label))

    # faces = get_subset(faces, subset_size)
    # labels = get_subset(labels, subset_size)
    faces = np.array(faces)
    labels = np.array(labels)
    print(labels)
    
    return faces, labels

# Get total number of images
total_images = sum(1 for _ in open(os.path.join(preprocessed_dir, 'labels.txt')))

# if subset_size != 0:
#     total_images = subset_size

# Load and train in chunks
for i in range(0, total_images, chunk_size):
    print(f"Processing chunk {i // chunk_size + 1}...")
    faces, labels = load_chunk(i, min(i + chunk_size, total_images))
    
    # Using train on the first chunk, update for subsequent ones
    if i == 0:
        face_recognizer.train(faces, labels)
    else:
        face_recognizer.update(faces, labels)
        
    print(f"Chunk {i // chunk_size + 1} processed.")

# Save the trained model
face_recognizer.save(model_save_path)

print(f"Model trained and saved at {model_save_path}")
