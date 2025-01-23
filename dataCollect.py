import cv2
from mtcnn import MTCNN
import os
import sys

def collect_face_data(person_name, roll_number, output_dir='data', max_images=150):
    """
    Collect face data and save in a structured directory format: data/PersonName_RollNumber/.

    Args:
        person_name (str): The name of the person for the dataset.
        roll_number (str): The roll number of the person for unique identification.
        output_dir (str): The root directory to save data.
        max_images (int): Number of face images to collect.
    """
    # Initialize the MTCNN detector
    detector = MTCNN()
    
    # Create a directory for the person with name and roll number
    person_dir = os.path.join(output_dir, f"{person_name}")
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    # Open webcam
    cap = cv2.VideoCapture(1)  # Change to 1 if using an external webcam
    img_id = 0

    while img_id < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Convert the frame to RGB as required by MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN
        detections = detector.detect_faces(rgb_frame)
        
        for face in detections:
            x, y, width, height = face['box']
            x, y = max(0, x), max(0, y)  # Prevent negative values
            cropped_face = frame[y:y+height, x:x+width]

            if cropped_face.size > 0:
                img_id += 1
                face_resized = cv2.resize(cropped_face, (200, 200))
                file_path = os.path.join(person_dir, f"{person_name}_{img_id}.jpg")
                cv2.imwrite(file_path, face_resized)
                
                # Display only the ID number in the label (no name)
                cv2.putText(frame, f"ID: {img_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        # Show the video feed with detections
        cv2.imshow("Face Collection", frame)

        # Press 'Enter' to stop manually
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collected {img_id} images for {person_name} (Roll No: {roll_number}) in {person_dir}")


if __name__ == "__main__":
    # Get name and roll number from the command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python dataCollect.py <person_name> <roll_number>")
        sys.exit(1)

    person_name = sys.argv[1]
    roll_number = sys.argv[2]

    # Call the function to collect face data
    collect_face_data(person_name=person_name, roll_number=roll_number, max_images=150)
