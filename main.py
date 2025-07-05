import cv2
import numpy as np
import pickle
import os
import dlib
import smtplib
import time
from email.message import EmailMessage
from ultralytics import YOLO

# Define base paths
BASE_DIR = "D:/Uniform_Detection"
FACE_DATA_DIR = os.path.join(BASE_DIR, "face_data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
VIOLATION_DIR = os.path.join(BASE_DIR, "violations")
VERIFIED_DIR = os.path.join(BASE_DIR, "verified")
UNAUTHORIZED_DIR = os.path.join(BASE_DIR, "unauthorized")

# Ensure required directories exist
for directory in [VIOLATION_DIR, VERIFIED_DIR, UNAUTHORIZED_DIR]:
    os.makedirs(directory, exist_ok=True)

# Load YOLO model for uniform detection
yolo_model_path = os.path.join(MODEL_DIR, "best.pt")
if not os.path.exists(yolo_model_path):
    raise FileNotFoundError("Error: YOLO model file not found.")

model = YOLO(yolo_model_path)

# Load stored face encodings
student_encodings = {}
if os.path.exists(FACE_DATA_DIR):
    for file in os.listdir(FACE_DATA_DIR):
        if file.endswith(".pkl"):
            with open(os.path.join(FACE_DATA_DIR, file), "rb") as f:
                data = pickle.load(f)
                student_encodings[data["name"]] = data
else:
    raise FileNotFoundError("Error: No student encodings found. Please register faces first.")

# Load dlib face detection models
sp_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
face_rec_path = os.path.join(MODEL_DIR, "dlib_face_recognition_resnet_model_v1.dat")

if not os.path.exists(sp_path) or not os.path.exists(face_rec_path):
    raise FileNotFoundError("Error: Required dlib models are missing.")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(sp_path)
facerec = dlib.face_recognition_model_v1(face_rec_path)

# Email Configuration
EMAIL_SENDER = "diet8632@gmail.com"
EMAIL_PASSWORD = "tcqhtqziihltfnmz"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

def send_email(to_email, subject, message, image_path=None):
    """Sends an email notification with optional image attachment."""
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = to_email
        msg.set_content(message)

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as img:
                msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename="Detected_Student.jpg")

        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)

        print(f"Email sent successfully to {to_email}")

    except Exception as e:
        print(f"Error: Failed to send email - {e}")

def recognize_student(image):
    """Identifies a student based on facial recognition."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if not faces:
        return None, None

    for face in faces:
        shape = sp(image, face)
        detected_embedding = np.array(facerec.compute_face_descriptor(image, shape))

        min_distance = float("inf")
        identified_student = None
        student_email = None

        for name, data in student_encodings.items():
            stored_encoding = np.array(data["encoding"])
            distance = np.linalg.norm(detected_embedding - stored_encoding)

            if distance < 0.55 and distance < min_distance:
                min_distance = distance
                identified_student = name
                student_email = data["email"]

        return identified_student, student_email

    return None, None

def capture_student_image():
    """Captures an image from the webcam."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Error: Unable to access webcam.")

    print("Camera is ON. Please align yourself properly...")
    time.sleep(3)  # Brief delay for proper alignment

    ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

    if not ret:
        raise RuntimeError("Error: Failed to capture image from webcam.")

    image_path = os.path.join(BASE_DIR, "captured_student.jpg")
    cv2.imwrite(image_path, frame)

    return frame, image_path

def detect_uniform(frame):
    """Detects uniform presence using YOLO."""
    results = model(frame)
    uniform_detected, non_uniform_detected = False, False

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == 1 and conf > 0.6:  # Class 1 = Uniform detected
                uniform_detected = True
            elif cls == 0 and conf > 0.6:  # Class 0 = Non-Uniform detected
                non_uniform_detected = True

    return uniform_detected, non_uniform_detected

def process_detection(frame, image_path):
    """Processes uniform detection and student identification."""
    student_name, student_email = recognize_student(frame)

    if student_name and student_email:
        print(f"Identified: {student_name} ({student_email})")

        uniform_detected, non_uniform_detected = detect_uniform(frame)

        if non_uniform_detected:
            print("Alert: Student detected without uniform. Sending fine email.")
            violation_path = os.path.join(VIOLATION_DIR, f"{student_name}.jpg")
            cv2.imwrite(violation_path, frame)

            subject = "Uniform Violation Fine"
            message = (f"Dear {student_name},\n\n"
                       "You have been detected without a uniform. A fine of RS.100/- has been imposed.\n\n"
                       "Best regards,\nAdministration")

            send_email(student_email, subject, message, violation_path)

        elif uniform_detected:
            print("Attendance marked successfully.")
            verified_path = os.path.join(VERIFIED_DIR, f"{student_name}.jpg")
            cv2.imwrite(verified_path, frame)

            subject = "Attendance Marked Successfully"
            message = (f"Dear {student_name},\n\n"
                       "Your attendance has been successfully marked as you are in proper uniform.\n\n"
                       "Best regards,\nAdministration")

            send_email(student_email, subject, message, verified_path)

    else:
        print("Warning: No recognized student detected. Marking as unauthorized entry.")
        unauthorized_path = os.path.join(UNAUTHORIZED_DIR, "unknown.jpg")
        cv2.imwrite(unauthorized_path, frame)

if __name__ == "__main__":
    try:
        captured_frame, image_path = capture_student_image()
        process_detection(captured_frame, image_path)
        print("System completed detection successfully.")

    except Exception as error:
        print(f"System encountered an error: {error}")