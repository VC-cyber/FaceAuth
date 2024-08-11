import cv2
import numpy as np
from model import getRecognitionModel, getFaceCascade
import os

def predict_label(image_path, model_path):
    """
    Predicts the label for a given image using a pre-trained face recognition model.
    
    Parameters:
    image_path (str): Path to the image to be predicted.
    model_path (str): Path to the pre-trained model file.
    
    Returns:
    label (int): The predicted label.
    confidence (float): The confidence of the prediction.
    """
    # Initialize the face recognizer
    face_recognizer = getRecognitionModel()
    face_cascade = getFaceCascade()
    
    # Load the trained model
    face_recognizer.read(model_path)
    
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at '{image_path}' could not be loaded.")
    
    # Resize image to match training data size
    #resized_image = cv2.resize(image, (100, 100))  # Use the same dimensions as used in training
    
    (x, y, w, h) = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_roi = image[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (100, 100))
    label, confidence = face_recognizer.predict(face_roi)
    return label, confidence
            
           
    # Predict the label
    
    

# Example usage

def main():
    image_path = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/1/'  # Path to the folder containing images
    model_path = '/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Weights/face_recognizer_model.yml'
    
    for image_name in os.listdir(image_path):
        try:
            # Construct the full image path
            full_image_path = os.path.join(image_path, image_name)
            
            print(f"Processing image: {full_image_path}")
            # Call your prediction function
            label, confidence = predict_label(full_image_path, model_path)
            
            print(f"Predicted Label: {label}")
            print(f"Confidence: {confidence}")
            
            # Print result based on label
            if label == 1:
                print("Accepted")
            else:
                print("Retrying")
        
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

    