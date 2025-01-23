import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.models import load_model

# Load trained model and label map
model = load_model('face_recognition_model.h5')
label_map = np.load('label_map.npy', allow_pickle=True).item()

# Initialize MTCNN face detector
detector = MTCNN()

# Initialize webcam
cap = cv2.VideoCapture(1)

# Set a confidence threshold for face recognition (e.g., 0.6)
confidence_threshold = 0.9

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    # Convert frame to RGB as required by MTCNN
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the frame
    detections = detector.detect_faces(rgb_frame)
    
    for face in detections:
        x, y, width, height = face['box']
        x, y = max(0, x), max(0, y)  # Prevent negative values
        cropped_face = frame[y:y+height, x:x+width]
        
        if cropped_face.size > 0:
            # Resize the cropped face to the input size of the model
            face_resized = cv2.resize(cropped_face, (200, 200))
            face_resized = face_resized / 255.0  # Normalize
            face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
            
            # Predict the class (person) of the detected face
            predictions = model.predict(face_resized)
            predicted_label = np.argmax(predictions, axis=1)[0]
            predicted_confidence = np.max(predictions, axis=1)[0]  # Confidence of the prediction
            
            # Check if confidence is above the threshold
            if predicted_confidence >= confidence_threshold:
                predicted_name = list(label_map.keys())[list(label_map.values()).index(predicted_label)]
            else:
                predicted_name = "Unknown"  # If confidence is too low, classify as "Unknown"
            
            # Draw bounding box and label around the face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, predicted_name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the resulting frame with face recognition results
    cv2.imshow("Face Recognition", frame)
    
    # Press 'Enter' to stop manually
    if cv2.waitKey(1) & 0xFF == 13:
            break

cap.release()
cv2.destroyAllWindows()
