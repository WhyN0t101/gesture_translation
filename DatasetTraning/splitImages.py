import os
import shutil
import random

# Define paths to your dataset folders
dataset_path = r'PATHTOImages'
train_path = r'PathToTraining'
val_path = r'PathToValidation'

# Create training and validation directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Define the percentage split for training and validation
train_split = 0.8  # 80% for training, 20% for validation

# Loop through each class folder in the dataset
for class_folder in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, class_folder)
    
    # Create class directories in the training and validation sets
    os.makedirs(os.path.join(train_path, class_folder), exist_ok=True)
    os.makedirs(os.path.join(val_path, class_folder), exist_ok=True)
    
    # Get list of image files in the class folder
    images = os.listdir(class_path)
    
    # Shuffle the list of images
    random.shuffle(images)
    
    # Calculate the number of images for training and validation
    num_train = int(len(images) * train_split)
    
    # Move images to training set
    for img in images[:num_train]:
        src = os.path.join(class_path, img)
        dest = os.path.join(train_path, class_folder, img)
        shutil.copy(src, dest)
    
    # Move images to validation set
    for img in images[num_train:]:
        src = os.path.join(class_path, img)
        dest = os.path.join(val_path, class_folder, img)
        shutil.copy(src, dest)

print("Dataset split into training and validation sets successfully.")
