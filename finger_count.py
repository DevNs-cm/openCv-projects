import cv2
import numpy as np

# Function to count fingers
def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        if defects is not None:
            cnt = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Calculate lengths
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))

                # Calculate angle
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

                # Filter defects
                if angle <= 90 and d > 3000:  # Reduced threshold for defect depth
                    cnt += 1
                    cv2.circle(drawing, far, 5, [0, 0, 255], -1)

            return cnt + 1 if cnt > 0 else 0
    return 0

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image

    # Define ROI
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range (adjusted for better detection)
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    # Threshold the HSV image to get skin
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological operations to remove noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 3000:
            # Draw contour
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # Count fingers
            fingers = count_fingers(max_contour, roi)

            # Display finger count
            cv2.putText(frame, f'Fingers: {fingers}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Finger Count', frame)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
