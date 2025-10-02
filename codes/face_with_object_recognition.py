# Import required libraries
import os  # For running system commands
import cv2  # OpenCV for image processing
import numpy as np  # For numerical computations
import face_recognition  # For facial recognition features
from picamera2 import Picamera2  # Camera interface for Raspberry Pi
import time  # To measure FPS and time-based events
import pickle  # To load pre-trained face encodings
import torch  # PyTorch used to load YOLOv5 model

# Load known face encodings from disk
print("[INFO] Loading face encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())  # Deserialize the pickle file
known_face_encodings = data["encodings"]  # Face vectors
known_face_names = data["names"]  # Corresponding names

# Load YOLOv5 object detection model from Ultralytics
print("[INFO] Loading YOLOv5 object detection model...")
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize and configure PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1280, 720)}))
picam2.start()

# Scale factor to reduce frame size for faster face recognition
cv_scaler = 2

# Initialize variables for tracking and performance
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0
last_face_spoken = ""  # Last recognized face announced
last_object_spoken = ""  # Last object announced

# Function to speak a given message using eSpeak
def speak(message):
    os.system(f'espeak "{message}"')  # Call eSpeak with message

# Function to process face recognition in a video frame
def process_faces(frame):
    global face_locations, face_encodings, face_names, last_face_spoken
    
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    
    # Convert image from BGR (OpenCV) to RGB (face_recognition)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and get face encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_names = []  # Reset the list of face names
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_names.append(name)

        # Speak name only if it's different from the last spoken one
        if name != last_face_spoken:
            sentence = f"{name} is coming your way" if name != "Unknown" else "Someone is coming your way"
            speak(sentence)
            last_face_spoken = name

# Function to draw face bounding boxes and names
def draw_faces(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale coordinates back to original frame size
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        # Draw the face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 2)
        
        # Draw label background
        cv2.rectangle(frame, (left, top - 30), (right, top), (244, 42, 3), cv2.FILLED)
        
        # Add text with the name
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 0.6, (255, 255, 255), 1)

# Function to detect objects using YOLOv5 and speak them out
def detect_objects(frame):
    global last_object_spoken
    results = yolo_model(frame)  # Run YOLO model on frame
    objects = results.pandas().xyxy[0]  # Get detections as pandas DataFrame
    spoken = False  # To ensure only one object is spoken per frame

    for index, row in objects.iterrows():
        label = row['name']  # Object class name
        conf = row['confidence']  # Confidence score
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Speak only the first new object detected in this frame
        if not spoken and label != last_object_spoken:
            sentence = f"{label} is in front of you"
            speak(sentence)
            last_object_spoken = label
            spoken = True

# Function to calculate and return current FPS
def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

# Main loop to run detection and recognition continuously
while True:
    frame = picam2.capture_array()  # Capture current frame from PiCamera
    
    process_faces(frame)  # Detect and recognize faces
    detect_objects(frame)  # Detect and announce objects
    draw_faces(frame)  # Draw rectangles and names for faces

    current_fps = calculate_fps()  # Measure performance

    # Display FPS on screen
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    
    # Show the video feed with annotations
    cv2.imshow("Face & Object Detection", frame)
    
    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup when exiting the loop
cv2.destroyAllWindows()
picam2.stop()
