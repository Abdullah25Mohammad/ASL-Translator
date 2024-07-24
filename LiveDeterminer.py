import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import mediapipe as mp
import numpy as np
from tf_keras.models import load_model
import time




model = load_model('smnist.h5')

cap = cv2.VideoCapture(0)

_, frame = cap.read()
h, w, c = frame.shape

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def clear():
    if os.name == 'nt':
        _ = os.system('cls')
    else:
        _ = os.system('clear')
        
def analyze_frame(frame):
    img = cv2.resize(frame, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model(img)

    clear()

    guess = np.argmax(pred)
    confidence = pred[0][guess] * 100
    
    return letters[guess], confidence



mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils



while True:
    _, frame = cap.read()
    # frame = cv2.flip(frame, 1)

    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        clear()
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Crop frame to focus hand only
    if results.multi_hand_landmarks:
        # for hand_landmarks in results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
        x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
        y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
        y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

        max_diff = max(x_max - x_min, y_max - y_min)

        k = 1.25

        center = ((x_min + x_max) // 2, (y_min + y_max) // 2)

        x_min = max(0, center[0] - int(max_diff/2 * k))
        x_max = min(w, center[0] + int(max_diff/2 * k))

        y_min = max(0, center[1] - int(max_diff/2 * k))
        y_max = min(h, center[1] + int(max_diff/2 * k))
        
        # mp_drawing.draw_landmarks(frame, hand_landmarks, mphands.HAND_CONNECTIONS)

        analysis_frame = frame[y_min:y_max, x_min:x_max]
        
        letter, confidence = analyze_frame(analysis_frame)


        # put bounding box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # put text
        cv2.putText(frame, f"Letter: {letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            
    cv2.imshow('Frame', frame)

    
    
    
