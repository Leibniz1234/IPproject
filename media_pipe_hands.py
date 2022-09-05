import mediapipe as mp
import cv2
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5,
                       min_tracking_confidence=0.5, static_image_mode=False)


def get_signal_from_landmark(landmarks, joint_num):
    signal = []
    pos = []
    for i in landmarks:
        pos.append(i.landmark[joint_num].x)
        pos.append(i.landmark[joint_num].y)
        pos.append(i.landmark[joint_num].z)
        signal.append(pos)
        pos = []

    return signal


def get_pos_of_joint_from_video(cap, joint_num, coordinate_type="multi_hand_landmarks"):
    cap = cv2.VideoCapture(cap)

    pos = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        # image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            if coordinate_type == "multi_hand_landmarks":
                pos.append(results.multi_hand_landmarks[0])
            elif coordinate_type == "multi_hand_world_landmarks":
                pos.append(results.multi_hand_world_landmarks[0])

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()

    return get_signal_from_landmark(pos, joint_num)


def get_dummy_signal(mediapipe_signal):
    signal = []
    for i in range(int(len(mediapipe_signal) / 2)):
        signal.append([mediapipe_signal[2 * i + 1][0] + np.random.normal(scale=0.01),
                       mediapipe_signal[2 * i + 1][1] + np.random.normal(scale=0.01),
                       mediapipe_signal[2 * i + 1][2] + np.random.normal(scale=0.01)])

    return signal
