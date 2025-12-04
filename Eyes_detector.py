import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye_landmarks):
    # Compute vertical distances
    A = math.dist(eye_landmarks[1], eye_landmarks[5])
    B = math.dist(eye_landmarks[2], eye_landmarks[4])
    # Compute horizontal distance
    C = math.dist(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for eye closed
EAR_THRESHOLD = 0.25

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks (optional)
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                   mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                   mp_draw.DrawingSpec(color=(0,0,255), thickness=1))

            # Get landmarks
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # Eye landmarks for right and left eyes (MediaPipe indices)
            left_eye_idx = [33, 160, 158, 133, 153, 144]    # right eye in mirror image
            right_eye_idx = [362, 385, 387, 263, 373, 380]  # left eye in mirror image

            left_eye = [landmarks[i] for i in left_eye_idx]
            right_eye = [landmarks[i] for i in right_eye_idx]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear_avg = (left_ear + right_ear) / 2.0

            if ear_avg < EAR_THRESHOLD:
                status = "Eyes Closed"
                color = (0,0,255)
            else:
                status = "Eyes Open"
                color = (0,255,0)

            cv2.putText(frame, status, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Eye Open/Close Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
