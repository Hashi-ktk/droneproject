from flask import Blueprint, Response, render_template
from djitellopy import Tello
import cv2
import mediapipe as mp
import threading
import logging
import time

hand_gesture_control = Blueprint('hand_gesture_control', __name__)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

gesture = 'Unknown'
tello = None  # Initialize tello as None initially


def init_tello():
    global tello
    tello = Tello()
    tello.LOGGER.setLevel(logging.ERROR)  # Ignore INFO from Tello
    tello.connect()
    print("Tello battery:", tello.get_battery())
    tello.streamon()
    tello.takeoff()
    time.sleep(2)
    tello.move_up(80)
    return tello


def hand_detection(tello):
    global gesture
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.flip(frame, 1)
        result = hands.process(frame)
        frame_height, frame_width, _ = frame.shape
        my_hand = []

        if result.multi_hand_landmarks:
            for handlms, handside in zip(result.multi_hand_landmarks, result.multi_handedness):
                if handside.classification[0].label == 'Right':
                    continue

                mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS,
                                      mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                                      mp.solutions.drawing_styles.get_default_hand_connections_style())

                for i, landmark in enumerate(handlms.landmark):
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    my_hand.append((x, y))

                finger_on = []
                if my_hand[4][0] > my_hand[2][0]:
                    finger_on.append(1)
                else:
                    finger_on.append(0)
                for i in range(1, 5):
                    if my_hand[4 + i * 4][1] < my_hand[2 + i * 4][1]:
                        finger_on.append(1)
                    else:
                        finger_on.append(0)

                gesture = 'Unknown'
                if sum(finger_on) == 0:
                    gesture = 'Stop'
                elif sum(finger_on) == 5:
                    gesture = 'Land'
                elif sum(finger_on) == 1:
                    if finger_on[0] == 1:
                        gesture = 'Right'
                    elif finger_on[4] == 1:
                        gesture = 'Left'
                    elif finger_on[1] == 1:
                        gesture = 'Up'
                elif sum(finger_on) == 2:
                    if finger_on[0] == finger_on[1] == 1:
                        gesture = 'Down'
                    elif finger_on[1] == finger_on[2] == 1:
                        gesture = 'Come'
                elif sum(finger_on) == 3 and finger_on[1] == finger_on[2] == finger_on[3] == 1:
                    gesture = 'Away'


@hand_gesture_control.route('/handgesture_video_feed')
def handgesture_video_feed():
    global tello
    def generate_frames():
        global gesture
        while True:
            frame = tello.get_frame_read().frame
            cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@hand_gesture_control.route('/connect_for_handgesture')
def connect_for_handgesture():
    try:
        global tello
        tello = init_tello()
        video_thread = threading.Thread(target=hand_detection, args=(tello,), daemon=True)
        video_thread.start()
        return render_template('handgesture.html')
    except Exception as e:
        print(f"Error during connect: {e}")
        return "Failed to connect to drone.", 500

@hand_gesture_control.route('/stop_handgesture')
def stop_handgesture():
    try:
        global tello
        tello.streamoff()
        tello.land()
        return render_template('streaming.html')
    except Exception as e:
        print(f"Error during disconnect: {e}")
        return "Failed to stop tracking and land the drone.", 500
