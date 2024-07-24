import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import cv2
import mediapipe as mp
import numpy as np



cap = cv2.VideoCapture(0)

_, frame = cap.read()
h, w, c = frame.shape


mphands = mp.solutions.hands
hands = mphands.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)



    # make an empty frame
    # frame = np.zeros((h, w, c), np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mphands.HAND_CONNECTIONS)

    
    cv2.imshow('Frame', frame)

    hand1 = np.zeros((21, 2))
    hand2 = np.zeros((21, 2))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks
        hand1 = np.array([(landmark.x, landmark.y) for landmark in hand_landmarks[0].landmark])
        
        if len(hand_landmarks) > 1:
            hand2 = np.array([(landmark.x, landmark.y) for landmark in hand_landmarks[1].landmark])

    print(hand1)
    print(hand2)




cap.release()