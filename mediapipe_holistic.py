import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def get_hand_landmark_from_video(cap):
    signal = []
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]:
                x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x
                y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y
                z = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z
                landmark = [x, y, z]
                signal.append(landmark)
            else:
                x = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x
                y = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y
                z = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z
                landmark = [x, y, z]
                signal.append(landmark)

            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    return signal
