# Import required libraries
import os
from imutils import paths             # Helps to get list of image paths in a directory
import face_recognition               # Library for face detection and encoding
import pickle                         # For saving encodings to a file
import cv2                            # OpenCV for reading and processing images

# Inform the user that face processing has started
print("[INFO] start processing faces...")

# Get list of image file paths from the 'dataset' directory
imagePaths = list(paths.list_images("dataset"))

# Lists to store the face encodings and corresponding names
knownEncodings = []
knownNames = []

# Loop through each image in the dataset
for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    
    # Extract the person's name from the directory name (assumes dataset/person_name/image.jpg)
    name = imagePath.split(os.path.sep)[-2]
    
    # Load the image and convert it from BGR (OpenCV default) to RGB (face_recognition requirement)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect face locations in the image using HOG model (faster, CPU-based)
    boxes = face_recognition.face_locations(rgb, model="hog")
    
    # Compute the 128-d face encodings for the detected faces
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    # Loop over each encoding found in the image
    for encoding in encodings:
        knownEncodings.append(encoding)  # Store the encoding
        knownNames.append(name)          # Store the name of the person

# Inform the user that data is being serialized
print("[INFO] serializing encodings...")

# Create a dictionary containing encodings and corresponding names
data = {"encodings": knownEncodings, "names": knownNames}

# Save the dictionary to a file using pickle
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Training complete. Encodings saved to 'encodings.pickle'")
