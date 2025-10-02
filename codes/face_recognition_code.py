# Import required libraries
import face_recognition  # For face detection and recognition
import cv2  # OpenCV for image processing and display
import numpy as np  # For numerical operations
from picamera2 import Picamera2  # PiCamera2 for accessing Raspberry Pi camera
import time  # For time-related operations
import pickle  # For loading saved face encoding data

# Load pre-trained face encodings from a file
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())  # Deserialize the pickled data
known_face_encodings = data["encodings"]  # Known face features
known_face_names = data["names"]  # Corresponding names

# Initialize and configure the PiCamera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()  # Start capturing

# Define initial variables
cv_scaler = 5  # Scaling factor to reduce image size and speed up processing

face_locations = []  # Detected face locations in the frame
face_encodings = []  # Encoded features of detected faces
face_names = []  # Names corresponding to detected faces
frame_count = 0  # Frame counter for FPS calculation
start_time = time.time()  # Start time for FPS calculation
fps = 0  # Frames per second value

# Function to process a video frame and recognize faces
def process_frame(frame):
    global face_locations, face_encodings, face_names
    
    # Resize frame to speed up processing
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    # Convert color from BGR (OpenCV default) to RGB (used by face_recognition)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and generate face encodings in the resized frame
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []  # Reset face names list
    for face_encoding in face_encodings:
        # Compare current face encoding with known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match
        
        # Find the closest match based on the smallest distance
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:  # Confirm if it's an actual match
            name = known_face_names[best_match_index]
        face_names.append(name)  # Append matched or default name
    
    return frame  # Return the original (non-resized) frame

# Function to draw rectangles and labels on detected faces
def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale face locations back to original frame size
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        # Draw a filled rectangle above the face for label
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        
        # Add name text inside the label rectangle
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    
    return frame  # Return the frame with drawings

# Function to calculate and update Frames Per Second (FPS)
def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1  # Count this frame
    elapsed_time = time.time() - start_time  # Time since last FPS update
    if elapsed_time > 1:  # Update every second
        fps = frame_count / elapsed_time
        frame_count = 0  # Reset frame count
        start_time = time.time()  # Reset start time
    return fps  # Return calculated FPS

# Main loop: runs continuously to capture, process, and display frames
while True:
    frame = picam2.capture_array()  # Capture a frame from the camera
    
    processed_frame = process_frame(frame)  # Detect and recognize faces
    
    display_frame = draw_results(processed_frame)  # Annotate faces with boxes and names
    
    current_fps = calculate_fps()  # Get FPS for performance monitoring
    
    # Display the FPS value on the video frame
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the final frame with annotations in a window
    cv2.imshow('Video', display_frame)
    
    # Exit the loop if the user presses the 'q' key
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup: close camera and window properly
cv2.destroyAllWindows()
picam2.stop()
