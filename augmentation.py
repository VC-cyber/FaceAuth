import os
from PIL import Image
import shutil 
from torchvision import transforms

# Define the directory containing your original images
image_folder = "/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/1"

# if os.path.exists(image_folder):
#     shutil.rmtree(image_folder)
#     print(f"Directory '{image_folder}' has been deleted.")

# os.makedirs(image_folder, exist_ok=True)
augment_folder = "/Users/venkat/Desktop/UCLA_CS/Summer_projects/FaceAuth/Data/1/augmentedImages"

# Define a list of augmentations that make sense for face images
augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with a probability of 0.5
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly change brightness, contrast, etc.
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Random translation and scaling
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Apply Gaussian blur
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # Random crop and resize
])

def augment_image(image, num_augments=10):
    """Generates multiple augmented versions of an input image."""
    augmented_images = []
    for _ in range(num_augments):
        augmented_image = augmentations(image)
        augmented_images.append(augmented_image)
    return augmented_images

# Iterate over all images in the folder
all_augmented_images = []

for filename in os.listdir(image_folder):
    print(f"Augmenting {filename}...")
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):  # Adjust the extensions as needed
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        
        # Apply augmentations
        augmented_images = augment_image(image)
        all_augmented_images.extend(augmented_images)
        
        # Optionally save augmented images
        base_filename = os.path.splitext(filename)[0]
        for idx, aug_img in enumerate(augmented_images):
            aug_img.save(os.path.join(augment_folder, f'{base_filename}_augmented_{idx}.jpg'))

print(f"Total augmented images: {len(all_augmented_images)}")
