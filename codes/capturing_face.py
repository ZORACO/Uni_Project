# Import necessary libraries
import cv2                    # OpenCV for image processing and display
import os                     # OS module to work with directories and file paths
from datetime import datetime # To generate timestamp for image filenames
from picamera2 import Picamera2 # To control the Raspberry Pi Camera
import time                   # To introduce delays

# Define the name of the person whose photos are being captured
PERSON_NAME = "Xyz"

# Function to create a dataset folder and a subfolder for the person
def create_folder(name):
    dataset_folder = "dataset"  # Root folder for dataset
    if not os.path.exists(dataset_folder):  # Check if it exists
        os.makedirs(dataset_folder)         # Create if it doesn't

    person_folder = os.path.join(dataset_folder, name)  # Subfolder for the specific person
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)         # Create person folder if it doesn't exist
    return person_folder                   # Return the path for saving photos

# Function to capture photos using Raspberry Pi Camera
def capture_photos(name):
    folder = create_folder(name)  # Ensure directory is set up

    # Initialize and configure the Pi Camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": 'XRGB8888', "size": (640, 480)}))  # Set resolution and format
    picam2.start()  # Start camera preview

    time.sleep(2)  # Allow camera to warm up

    photo_count = 0  # Counter for saved photos

    print(f"Taking photos for {name}. Press SPACE to capture, 'q' to quit.")

    while True:
        frame = picam2.capture_array()  # Capture current frame from the camera

        cv2.imshow('Capture', frame)    # Show the frame in a window

        key = cv2.waitKey(1) & 0xFF     # Wait for a key press (non-blocking)

        if key == ord(' '):  # If spacebar is pressed
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
            filename = f"{name}_{timestamp}.jpg"  # Create a filename
            filepath = os.path.join(folder, filename)  # Full path to save image
            cv2.imwrite(filepath, frame)  # Save the image
            print(f"Photo {photo_count} saved: {filepath}")

        elif key == ord('q'):  # If 'q' is pressed, exit loop
            break

    # Cleanup: close window and stop the camera
    cv2.destroyAllWindows()
    picam2.stop()
    print(f"Photo capture completed. {photo_count} photos saved for {name}.")

# Main execution
if __name__ == "__main__":
    capture_photos(PERSON_NAME)  # Start the photo capturing process
