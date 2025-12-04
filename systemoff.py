import cv2
import mediapipe as mp
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Count number of fingers open
def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]  # finger tips
    open_count = 0

    for tip in tips:
        # if finger tip is above the PIP joint → finger is open
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            open_count += 1

    return open_count


cap = cv2.VideoCapture(0)

armed = False  # To detect 2-finger trigger

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                finger_count = count_fingers(hand_landmarks)

                # 1️⃣ Step 1: Detect 2 fingers
                if finger_count == 2 and not armed:
                    armed = True
                    cv2.putText(frame, "2 Fingers Detected - READY", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                # 2️⃣ Step 2: If already armed → detect full open hand
                elif armed and finger_count >= 4:
                    cv2.putText(frame, "Full Hand Detected! Shutdown...", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    os.system("shutdown /s /t 1")
                    time.sleep(2)
                    break

                # Show current finger count for debugging
                cv2.putText(frame, f"Fingers: {finger_count}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Hand Shutdown Control", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
