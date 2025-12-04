import cv2
import mediapipe as mp
import numpy as np
import time
import platform

# Try to import pycaw (Windows). If unavailable, fall back to no-system-volume mode.
USE_PYCAW = False
if platform.system() == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volRange = volume.GetVolumeRange()  # (minVol, maxVol, step)
        minVol, maxVol = volRange[0], volRange[1]
        USE_PYCAW = True
        print("pycaw initialized. Volume control available (Windows).")
    except Exception as e:
        print("pycaw available but failed to initialize:", e)
        USE_PYCAW = False
else:
    print("Non-Windows OS detected — system volume control via pycaw is not available.")
    USE_PYCAW = False

# MediaPipe hands initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Camera open (tries indices 0..4)
cap = None
for i in range(5):
    test = cv2.VideoCapture(i, cv2.CAP_DSHOW if platform.system() == "Windows" else 0)
    time.sleep(0.2)
    if test.isOpened():
        cap = test
        print(f"Opened camera index {i}")
        break
    test.release()

if cap is None or not cap.isOpened():
    print("ERROR: Could not open camera. Check camera device or permissions.")
    raise SystemExit("Camera not available")

# Smoothing parameters
prev_vol = None
SMOOTHING = 0.2   # 0..1 lower = smoother (less jitter)
LENGTH_MIN = 20   # minimum finger distance mapped to min volume
LENGTH_MAX = 200  # maximum finger distance mapped to max volume

# frame loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed — exiting.")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)

    info_text = "Show thumb & index to control volume"
    vol_percent_display = None

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark

        # convert landmarks to pixel coordinates
        lm_list = [(int(pt.x * w), int(pt.y * h)) for pt in lm]

        # Ensure indices 4 and 8 exist
        if len(lm_list) > 8:
            x1, y1 = lm_list[4]   # Thumb tip
            x2, y2 = lm_list[8]   # Index tip

            # Draw landmarks / connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Visual helpers
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (x1, y1), 8, (255, 0, 255), -1)
            cv2.circle(frame, (x2, y2), 8, (255, 0, 255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

            # Distance and mapping
            length = np.hypot(x2 - x1, y2 - y1)  # Euclidean
            # clamp length to avoid out-of-range values
            length_clamped = float(np.clip(length, LENGTH_MIN, LENGTH_MAX))

            # Map length to percentage 0..100
            vol_percent = np.interp(length_clamped, [LENGTH_MIN, LENGTH_MAX], [0, 100])
            vol_percent_display = int(vol_percent)

            # Map to system volume range if available
            if USE_PYCAW:
                # Map to pycaw range (minVol..maxVol)
                vol_mapped = np.interp(length_clamped, [LENGTH_MIN, LENGTH_MAX], [minVol, maxVol])

                # Smooth the volume changes
                if prev_vol is None:
                    smooth_vol = vol_mapped
                else:
                    smooth_vol = prev_vol + SMOOTHING * (vol_mapped - prev_vol)

                try:
                    volume.SetMasterVolumeLevel(float(smooth_vol), None)
                    prev_vol = smooth_vol
                except Exception as e:
                    cv2.putText(frame, "Volume set failed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    print("Error setting volume:", e)
            else:
                # Not Windows / pycaw not available: don't change system volume
                prev_vol = None

            # Draw a volume bar on the frame (visual)
            vol_bar_y = int(np.interp(length_clamped, [LENGTH_MIN, LENGTH_MAX], [400, 150]))
            cv2.rectangle(frame, (50, 150), (85, 400), (200, 200, 200), 2)
            cv2.rectangle(frame, (50, vol_bar_y), (85, 400), (0, 200, 0), -1)
            cv2.putText(frame, f"{vol_percent_display if vol_percent_display is not None else 0}%", (40, 430),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

            # Mute indicator if very close
            if length < (LENGTH_MIN + 5):
                cv2.circle(frame, (cx, cy), 16, (0, 0, 255), -1)
                info_text = "MUTE (fingers very close)"
            else:
                info_text = "Adjusting volume"

        else:
            info_text = "Landmark indices missing"
    else:
        info_text = "No hand detected"

    # On-screen instructions & FPS
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if vol_percent_display is not None:
        cv2.putText(frame, f"Vol: {vol_percent_display}%", (w - 160, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Volume Control", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
