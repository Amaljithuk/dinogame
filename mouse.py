import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Smoothing factor (to reduce cursor jitter)
smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

# Open webcam
cap = cv2.VideoCapture(0)

def is_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]  # Thumb tip (Landmark 4)
    index_tip = hand_landmarks.landmark[8]  # Index tip (Landmark 8)
    distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.05  # Adjust threshold as needed

def is_right_click(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    distance = ((thumb_tip.x - pinky_tip.x)**2 + (thumb_tip.y - pinky_tip.y)**2)**0.5
    return distance < 0.05

def is_drag(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    middle_tip = hand_landmarks.landmark[12]
    distance = ((thumb_tip.x - middle_tip.x)**2 + (thumb_tip.y - middle_tip.y)**2)**0.5
    return distance < 0.05

while True:
    success, img = cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)  # Mirror the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip (Landmark 8)
            index_tip = hand_landmarks.landmark[8]
            h, w, _ = img.shape
            
            # Convert normalized coordinates to screen position
            mouse_x = int(index_tip.x * screen_width)
            mouse_y = int(index_tip.y * screen_height)
            
            # Smooth cursor movement
            curr_x = prev_x + (mouse_x - prev_x) / smoothening
            curr_y = prev_y + (mouse_y - prev_y) / smoothening
            
            # Move mouse
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            # Check gestures
            if is_pinch(hand_landmarks):
                pyautogui.click()  # Left-click
                cv2.putText(img, "LEFT CLICK", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_right_click(hand_landmarks):
                pyautogui.rightClick()  # Right-click
                cv2.putText(img, "RIGHT CLICK", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif is_drag(hand_landmarks):
                pyautogui.mouseDown()  # Start dragging
                cv2.putText(img, "DRAGGING", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                pyautogui.mouseUp()  # Stop dragging
            
            # Draw hand landmarks (optional)
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow("Gesture Mouse Controller", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()