import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time

# -----------------------------
# Configurations
# -----------------------------
MODEL_PATH = "yolov8n.pt"       # small, fast model
DETECTION_THRESHOLD = 0.4       # min confidence
FRAME_WIDTH = 720               # webcam frame width
FRAME_HEIGHT = 480              # webcam frame height
BRAND_TEXT = "Dev N Suman Vision AI"

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO(MODEL_PATH)

# -----------------------------
# MediaPipe Hand Setup
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

def count_fingers(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other 4 fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)

# -----------------------------
# Fancy box for attractive UI
# -----------------------------
def draw_fancy_box(img, box, label, conf, color=(0, 200, 0)):
    x1, y1, x2, y2 = box
    # Semi-transparent rectangle
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    # Text background
    txt = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - 22), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

# -----------------------------
# Webcam capture
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reduce frame size for processing → speed up
    input_frame = cv2.resize(frame, (FRAME_WIDTH//2, FRAME_HEIGHT//2))

    # -----------------------------
    # YOLO detection
    # -----------------------------
    results = model(input_frame, stream=True)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < DETECTION_THRESHOLD:
                continue
            cls = int(box.cls[0])
            label = r.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Scale back to original frame
            scale_x = FRAME_WIDTH / (FRAME_WIDTH//2)
            scale_y = FRAME_HEIGHT / (FRAME_HEIGHT//2)
            x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
            draw_fancy_box(frame, (x1, y1, x2, y2), label, conf)

    # -----------------------------
    # Hand detection + finger count
    # -----------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    if hand_results.multi_hand_landmarks:
        for hand_lm in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
            fingers = count_fingers(hand_lm)
            cv2.putText(frame, f"Fingers: {fingers}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,50,50), 3)

    # -----------------------------
    # Branding + FPS
    # -----------------------------
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time + 1e-8)
    prev_time = curr_time
    cv2.putText(frame, BRAND_TEXT, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50,200,255), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

    # -----------------------------
    # Display smaller window for comfort
    # -----------------------------
    display = cv2.resize(frame, (640, 480))
    cv2.imshow("Dev N Suman Detector", display)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
        break

cap.release()
cv2.destroyAllWindows()
