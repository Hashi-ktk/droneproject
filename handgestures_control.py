from flask import Blueprint, render_template, Response
import cv2
import mediapipe as mp
from djitellopy import Tello
from flask_login import login_required, current_user
import numpy as np
import logging
import threading
import time
import os

handgestures_control = Blueprint('handgestures_control', __name__)

# Width and height of the camera 360, 240
w, h = 360, 240

# Tello state variables
tello = None
takeoff = False
land = False
stop_gestures = False
recording = False
out = None

# MediaPipe hands detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Gesture variable
gesture = 'Unknown'

# Initialize Tello
def init_tello():
    try:
        tello = Tello()
        Tello.LOGGER.setLevel(logging.WARNING)
        tello.connect()
        print("Tello battery:", tello.get_battery())

        # Velocity values
        tello.for_back_velocity = 0
        tello.left_right_velocity = 0
        tello.up_down_velocity = 0
        tello.yaw_velocity = 0
        tello.speed = 0

        # Streaming
        tello.streamoff()
        tello.streamon()

        return tello
    except Exception as e:
        print(f"Error initializing Tello: {e}")
        return None

# Get frame on stream
def get_frame(tello, w=w, h=h):
    try:
        tello_frame = tello.get_frame_read().frame
        return cv2.resize(tello_frame, (w, h))
    except Exception as e:
        print(f"Error getting frame: {e}")
        return np.zeros((h, w, 3), dtype=np.uint8)

# Hand detection and gesture recognition
def hand_detection(tello):
    global gesture
    while True:
        if stop_gestures:
            break

        frame = get_frame(tello, w, h)
        frame = cv2.flip(frame, 1)
        result = hands.process(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]
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

        cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if recording and out is not None:
            out.write(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if gesture == 'Land':
            break

# Route handler for video stream
@handgestures_control.route('/handgestures_video_feed')
def handgestures_video_feed():
    global tello, takeoff, land, stop_gestures, recording, out
    stop_gestures = False
    tello = init_tello()

    def generate_frames():
        global tello, takeoff, land, stop_gestures, recording, out
        while True:
            if stop_gestures:
                if land:
                    tello.streamoff()
                    tello.land()
                    tello.end()
                    tello = None
                    land = False
                if recording and out is not None:
                    out.release()
                    recording = False
                break

            if not takeoff:
                try:
                    tello.takeoff()
                    tello.move_up(80)
                    takeoff = True
                    land = True
                    time.sleep(2.2)
                except Exception as e:
                    print(f"Error during takeoff: {e}")

            yield from hand_detection(tello)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@handgestures_control.route('/capture_image')
@login_required
def capture_image():
    global tello
    try:
        user = current_user.name
        img = get_frame(tello, w, h)
        user_dir = os.path.join('Images', user)
        os.makedirs(user_dir, exist_ok=True)
        i = 0
        while True:
            image_path = os.path.join(user_dir, f'image{i}.jpg')
            if not os.path.exists(image_path):
                cv2.imwrite(image_path, img)
                break
            i += 1
        return f"Image saved as {image_path}"
    except Exception as e:
        print(f"Error capturing image: {e}")
        return "Failed to capture image"

@handgestures_control.route('/start_recording')
@login_required
def start_recording():
    global recording, out
    try:
        user = current_user.name
        if not recording:
            user_dir = os.path.join('Videos', user)
            os.makedirs(user_dir, exist_ok=True)
            i = 0
            while True:
                video_path = os.path.join(user_dir, f'video{i}.mp4')
                if not os.path.exists(video_path):
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
                    recording = True
                    break
                i += 1
            return f"Recording started and will be saved as {video_path}"
        return "Already recording"
    except Exception as e:
        print(f"Error starting recording: {e}")
        return "Failed to start recording"

@handgestures_control.route('/stop_recording')
def stop_recording():
    global recording, out
    try:
        if recording and out is not None:
            out.release()
            recording = False
            return "Recording stopped and saved"
        return "No active recording to stop"
    except Exception as e:
        print(f"Error stopping recording: {e}")
        return "Failed to stop recording"

@handgestures_control.route('/stop_handgestures')
def stop_handgestures():
    global stop_gestures
    try:
        stop_gestures = True
        return render_template('profile.html')
    except Exception as e:
        print(f"Error stopping hand gestures: {e}")
        return "Failed to stop hand gestures"

@handgestures_control.route('/connect_to_handgestures')
def connect_to_handgestures():
    return render_template('handgestures.html')

@handgestures_control.route('/disconnect_to_handgestures')
def disconnect_to_handgestures():
    global stop_gestures
    try:
        stop_gestures = True
        return render_template('profile.html')
    except Exception as e:
        print(f"Error stopping hand gestures: {e}")
        return "Failed to stop hand gestures"