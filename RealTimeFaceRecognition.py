import os
import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Load the trained model and label map
model = load_model('face_recognition_model.h5')
label_map = np.load('label_map.npy', allow_pickle=True).item()

# Load the feature vectors for all persons
features_dir = "features"
feature_vectors = {}
for feature_file in os.listdir(features_dir):
    if feature_file.endswith('.npy'):
        person_name = feature_file.split('_')[0]
        feature_vector = np.load(os.path.join(features_dir, feature_file))
        feature_vectors[person_name] = feature_vector

# Function to preprocess the input image and extract features
def preprocess_and_extract_features(img):
    img_resized = cv2.resize(img, (200, 200))  # Resize to match model input size
    img_resized = img_resized / 255.0  # Normalize the image
    img_resized = np.expand_dims(img_resized, axis=0)  # Add batch dimension

    # Create an intermediate model to extract features (penultimate layer)
    intermediate_layer_model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    feature_vector = intermediate_layer_model.predict(img_resized)[0]
    return feature_vector

# Function to recognize the person from the captured image
def recognize_person(img, threshold=0.7):  # Increased threshold to 0.9 for stricter recognition
    feature_vector = preprocess_and_extract_features(img)

    # Calculate cosine similarity between the input image feature and all saved feature vectors
    similarities = {}
    for person_name, saved_feature in feature_vectors.items():
        similarity = cosine_similarity([feature_vector], [saved_feature])[0][0]
        print(f"Comparing with {person_name}: Similarity = {similarity}")  # Debug print
        similarities[person_name] = similarity

    # Find the person with the highest similarity score
    recognized_person = max(similarities, key=similarities.get)
    recognition_score = similarities[recognized_person]

    # If recognition score is below threshold, return 'Unknown'
    if recognition_score < threshold:
        recognized_person = 'Unknown'

    return recognized_person, recognition_score

# Initialize MTCNN detector for face detection
detector = MTCNN()

# Initialize the camera
cap = cv2.VideoCapture(1)

# Ensure camera is opened successfully
if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Detect faces using MTCNN
    faces = detector.detect_faces(frame)

    # Process each face in the frame
    for face in faces:
        # Extract the bounding box of the face
        x, y, w, h = face['box']

        # Crop the face region from the frame
        face_img = frame[y:y+h, x:x+w]

        # Recognize the person for the cropped face
        recognized_person, recognition_score = recognize_person(face_img)

        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the recognized person's name and confidence score
        cv2.putText(frame, f"Name: {recognized_person}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Confidence: {recognition_score:.2f}", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show the frame with detected faces and annotations
    cv2.imshow("Face Recognition", frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
