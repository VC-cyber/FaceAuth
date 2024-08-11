import cv2
import os
import numpy as np
from model import getFaceCascade, getRecognitionModel
import shutil

augmented = True
# Paths to your datasets
celeba_dataset_path = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/celeba/img_align_celeba'
if(augmented):
    my_dataset_path = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/1/augmentedImages'
else:
    my_dataset_path = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/1'

# Directory to save preprocessed images
preprocessed_dir = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/preprocessed_data'

#if exists delete the directory
if os.path.exists(preprocessed_dir):
    shutil.rmtree(preprocessed_dir)
    print(f"Directory '{preprocessed_dir}' has been deleted.")
# Create the directory if it does not exist
os.makedirs(preprocessed_dir, exist_ok=True)

# Haarcascade face detector
face_cascade = getFaceCascade()

subset_size = 300
# Function to preprocess images and save them
def preprocess_and_save_images(image_path, label, save_dir):
    maxIter = 0
    for i, filename in enumerate(os.listdir(image_path)):
        print(f"Preprocessing {filename}...")
        if(maxIter > subset_size and subset_size > 0):
            break
        if filename.endswith(".jpeg") or filename.endswith(".jpg")or filename.endswith(".png"):
            img_path = os.path.join(image_path, filename)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for j, (x, y, w, h) in enumerate(detected_faces):
                face_roi = gray[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (200, 200))
                face_filename = f"{label}_{i}_{j}.png"
                face_path = os.path.join(save_dir, face_filename)
                cv2.imwrite(face_path, face_roi)

                with open(os.path.join(save_dir, 'labels.txt'), 'a') as label_file:
                    label_file.write(f"{face_filename},{label}\n")
        maxIter += 1

# Preprocess and save CelebA dataset (label 0)
preprocess_and_save_images(celeba_dataset_path, label=0, save_dir=preprocessed_dir)

# Preprocess and save your dataset (label 1)
preprocess_and_save_images(my_dataset_path, label=1, save_dir=preprocessed_dir)

print("Preprocessed images and labels saved.")
