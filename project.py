import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import winsound

# Load Dlib's face detector and facial landmarks model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def play_alarm():
    winsound.Beep(1000, 1000)  # 1000Hz for 1000ms (1 second)
    time.sleep(0.2)  # Short pause
    winsound.Beep(1000, 1000)  # Second beep

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Eye landmark indices
LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]

# Thresholds
EAR_THRESHOLD = 0.25  # If below, eyes are considered closed
CLOSURE_TIME = 3.0     # Time in seconds to trigger alarm

cap = cv2.VideoCapture(0)  # Capture from webcam
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        left_eye = shape[LEFT_EYE_IDX]
        right_eye = shape[RIGHT_EYE_IDX]
        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Draw eyes on the frame
        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)

        # Display EAR value
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if avg_ear < EAR_THRESHOLD:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time >= CLOSURE_TIME:
                cv2.putText(frame, "ALERT! WAKE UP!", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                print("ALARM! Eyes closed for too long!")
                play_alarm()  # Play alarm sound
        else:
            start_time = None  # Reset timer when eyes are open

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
