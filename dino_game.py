import cv2
import mediapipe as mp
import numpy as np
import os
import json

# -----------------------------
# Score Handling
# -----------------------------
BEST_FILE = "best_score.json"

def load_best():
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE, "r") as f:
            return json.load(f).get("best", 0)
    return 0

def save_best(score):
    with open(BEST_FILE, "w") as f:
        json.dump({"best": score}, f)

best_score = load_best()

# -----------------------------
# Hand Detection
# -----------------------------
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
tip_ids = [4, 8, 12, 16, 20]

# -----------------------------
# Load Dino & Cactus Images
# -----------------------------
dino_img = cv2.imread("dino.png", cv2.IMREAD_UNCHANGED)
cactus_img = cv2.imread("cactus.png", cv2.IMREAD_UNCHANGED)

# Ensure 4 channels
if dino_img.shape[2] == 3:
    dino_img = cv2.cvtColor(dino_img, cv2.COLOR_BGR2BGRA)
if cactus_img.shape[2] == 3:
    cactus_img = cv2.cvtColor(cactus_img, cv2.COLOR_BGR2BGRA)

# Resize
dino_img = cv2.resize(dino_img, (70, 90))
cactus_img = cv2.resize(cactus_img, (60, 80))

# -----------------------------
# Safe PNG Overlay
# -----------------------------
def overlay_png(bg, png, x, y):
    h, w = png.shape[:2]
    if x < 0:
        png = png[:, -x:]
        w = png.shape[1]
        x = 0
    if y < 0:
        png = png[-y:, :]
        h = png.shape[0]
        y = 0
    if x + w > bg.shape[1]:
        w = bg.shape[1] - x
        png = png[:, :w]
    if y + h > bg.shape[0]:
        h = bg.shape[0] - y
        png = png[:h, :]
    if w <= 0 or h <= 0:
        return bg
    alpha = png[:, :, 3] / 255.0
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (1-alpha)*bg[y:y+h, x:x+w, c] + alpha*png[:h, :w, c]
    return bg

# -----------------------------
# Game Variables
# -----------------------------
dino_x, dino_y = 80, 330
dino_w, dino_h = 70, 90

cactus_x, cactus_y = 640, 340
cactus_w, cactus_h = 60, 80

jump = False
jump_vel = 0
gravity = 2
speed = 12
score = 0

# -----------------------------
# Start Camera
# -----------------------------

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # Remove solid background! Keep your hands visible
    # frame[:] = (235, 235, 235)  # <- REMOVE THIS

    # Draw ground line like Chrome Dino
    cv2.line(frame, (0, 420), (640,420), (0,0,0), 3)

    # (rest of code: hand detection, overlay Dino, cactus, score, collision)


    # -----------------------------
    # Hand Detection
    # -----------------------------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    hand_state = "No Hand"
    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Determine hand open/closed
            fingers = []
            lm = hand_landmarks.landmark
            for id in range(1,5):
                if lm[tip_ids[id]].y < lm[tip_ids[id]-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # Thumb
            if lm[tip_ids[0]].x < lm[tip_ids[0]-1].x:
                fingers.insert(0,1)
            else:
                fingers.insert(0,0)
            if sum(fingers) >= 3:
                hand_state = "Open"
            else:
                hand_state = "Closed"

    # -----------------------------
    # Jump Only if Hand Open
    # -----------------------------
    if hand_detected and hand_state=="Open" and not jump:
        jump = True
        jump_vel = -24

    # Jump physics
    if jump:
        dino_y += jump_vel
        jump_vel += gravity
        if dino_y >= 330:
            dino_y = 330
            jump = False

    # -----------------------------
    # Obstacle Movement
    # -----------------------------
    cactus_x -= speed
    if cactus_x < -80:
        cactus_x = 640
        score += 1
        speed += 0.7

    # -----------------------------
    # Draw Ground
    # -----------------------------
    cv2.line(frame, (0, 420), (640,420), (0,0,0), 3)

    # Draw Dino & Cactus
    frame = overlay_png(frame, dino_img, int(dino_x), int(dino_y))
    frame = overlay_png(frame, cactus_img, int(cactus_x), int(cactus_y))

    # -----------------------------
    # Collision Detection
    # -----------------------------
    dino_rect = (dino_x, dino_y, dino_x + dino_w, dino_y + dino_h)
    cactus_rect = (cactus_x, cactus_y, cactus_x + cactus_w, cactus_y + cactus_h)
    def coll(a,b):
        return not(a[2]<b[0] or a[0]>b[2] or a[3]<b[1] or a[1]>b[3])
    if coll(dino_rect,cactus_rect):
        if score > best_score:
            best_score = score
            save_best(best_score)
        cv2.putText(frame,"GAME OVER!",(200,200),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
        cv2.putText(frame,f"Score: {score}",(240,260),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.putText(frame,f"Best: {best_score}",(240,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        cv2.imshow("Chrome Dino", frame)
        cv2.waitKey(1500)
        # Reset
        score = 0
        speed = 12
        cactus_x = 640
        dino_y = 330
        jump = False
        continue

    # -----------------------------
    # Display Score & Hand State
    # -----------------------------
    cv2.putText(frame,f"Score: {score}",(500,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.putText(frame,f"Best: {best_score}",(500,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.putText(frame,f"Hand: {hand_state}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

    cv2.imshow("Chrome Dino", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
