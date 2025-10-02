# Face and Object Recognition on Raspberry Pi

This project demonstrates how to build a **real-time face and object recognition system** using a Raspberry Pi camera.  
It uses **OpenCV**, **face_recognition**, and **YOLOv5 (PyTorch)** for computer vision tasks, along with **eSpeak** for audio feedback.

---

## Features
- Capture and save images of a person to build a dataset.
- Train and load known face encodings for recognition.
- Perform real-time face recognition with labels and bounding boxes.
- Detect objects using YOLOv5 alongside face recognition.
- Provide **audio feedback** for detected people and objects.

---

## Files

### `capturing_face.py`
- Captures images of a person from the PiCamera.
- Saves images into a `dataset` folder for training purposes.
- Controls:  
  - Press **SPACE** → Capture a photo  
  - Press **q** → Quit

### `train_model.py` (encoding generator)
- Processes all images inside the `dataset/` directory.  
- Detects faces and extracts **128-d encodings** using `face_recognition`.  
- Saves encodings and names into a serialized file called **`encodings.pickle`**.  
- This file is required by the recognition scripts. 

### `face_recognition_code.py`
- Loads a pre-trained `encodings.pickle` file containing face encodings.
- Performs **real-time face recognition** using the Raspberry Pi camera.
- Draws bounding boxes and names around detected faces.
- Displays FPS for performance tracking.
- can get better fps by changing the value of cv_scaler but need to compromise with efficiency


### `face_with_object_recognition.py`
- Extends face recognition by also performing **object detection** with YOLOv5.
- Detects both faces and objects in the camera feed simultaneously.
- Provides **audio feedback** using `espeak`, announcing recognized people and objects.
- Displays FPS on screen.

---

## Requirements
- Raspberry Pi with PiCamera2 support
- Python 3.8+
- Required Python libraries (see `requirements.txt`)
- `espeak` installed for audio announcements:
  ```bash
  sudo apt-get install espeak


## To install the requirements you need to write in terminal 
       pip install -r requirements.txt
## to install in Raspberry pi you need to write in bash
     sudo apt-get install especk && -r requirements.txt
