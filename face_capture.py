import cv2
import numpy as np
import os
import pickle
import dlib

# Define base directories
BASE_DIR = "D:/Uniform_Detection"
FACE_DATA_DIR = os.path.join(BASE_DIR, "face_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Ensure the face data directory exists
os.makedirs(FACE_DATA_DIR, exist_ok=True)

# Load dlib models
detector = dlib.get_frontal_face_detector()
sp_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
face_rec_path = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

# Ensure model files exist before loading
if not os.path.exists(sp_path) or not os.path.exists(face_rec_path):
    raise FileNotFoundError("Required dlib models not found in the models directory.")

sp = dlib.shape_predictor(sp_path)
facerec = dlib.face_recognition_model_v1(face_rec_path)

def capture_and_encode_face(student_name, student_email):
    """Captures and encodes a student's face."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    print("Press 's' to capture and encode the face.")
    print("Press 'q' to quit without saving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        faces = detector(frame)
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and faces:
            shape = sp(frame, faces[0])
            face_descriptor = facerec.compute_face_descriptor(frame, shape)
            face_data = {"name": student_name, "email": student_email, "encoding": np.array(face_descriptor)}
            face_data_path = os.path.join(FACE_DATA_DIR, f"{student_name}.pkl")

            with open(face_data_path, "wb") as f:
                pickle.dump(face_data, f)

            print(f"Face encoded and saved for {student_name} ({student_email}) at {face_data_path}")
            break

        elif key == ord('q'):
            print("Capture aborted. No face saved.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    student_name = input("Enter Student Name: ").strip()
    student_email = input("Enter Student Email: ").strip()

    if student_name and student_email:
        capture_and_encode_face(student_name, student_email)
    else:
        print("Student name and email cannot be empty.")
