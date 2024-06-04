from flask import Blueprint, render_template, Response
from djitellopy import Tello
from flask_login import login_required, current_user
import cv2
import mediapipe as mp
import threading
import logging
import os
import time

handgestures_control = Blueprint('handgestures_control', __name__)

# Initialize MediaPipe for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

# Global variables
tello = None
gesture = 'Unknown'
fly = False
stop_tracking = False
recording = False
out = None

def init_tello():
    global tello
    if tello is None:
        tello = Tello()
        Tello.LOGGER.setLevel(logging.WARNING)
        tello.connect()
        print("Tello battery:", tello.get_battery())

        # Streaming
        tello.streamoff()
        tello.streamon()

def hand_detection():
    global gesture, stop_tracking, recording, out

    while True:
        if stop_tracking:
            tello.streamoff()
            tello.end()
            if recording and out is not None:
                out.release()
                recording = False
            break

        # Read the frame from Tello
        frame = tello.get_frame_read().frame
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
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

        if recording and out is not None:
            out.write(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def control_drone():
    global gesture, stop_tracking

    while not stop_tracking:
        hV = dV = vV = rV = 0
        if gesture == 'Land':
            break
        elif gesture == 'Stop' or gesture == 'Unknown':
            hV = dV = vV = rV = 0
        elif gesture == 'Right':
            hV = -15
        elif gesture == 'Left':
            hV = 15
        elif gesture == 'Up':
            vV = 20
        elif gesture == 'Down':
            vV = -20
        elif gesture == 'Come':
            dV = 15
        elif gesture == 'Away':
            dV = -15

        tello.send_rc_control(hV, dV, vV, rV)
        time.sleep(0.1)

    tello.land()
    tello.streamoff()
    print("Battery:", tello.get_battery())

@handgestures_control.route('/handgestures_video_feed')
def handgestures_video_feed():
    global stop_tracking
    stop_tracking = False

    init_tello()

    if not tello.is_flying:
        tello.takeoff()
        time.sleep(2)
        tello.move_up(80)

    threading.Thread(target=control_drone, daemon=True).start()

    return Response(hand_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@handgestures_control.route('/stop_handgestures')
def stop_handgestures():
    global stop_tracking
    stop_tracking = True
    return render_template('profile.html')

@handgestures_control.route('/connect_to_handgestures')
def connect_to_handgestures():
    return render_template('handgestures.html')

@handgestures_control.route('/disconnect_to_handgestures')
def disconnect_to_handgestures():
    global tello
    tello.streamoff()
    tello.end()
    return render_template('profile.html')

@handgestures_control.route('/capture_image')
@login_required
def capture_image():
    global tello
    user = current_user.name
    frame = tello.get_frame_read().frame
    img = cv2.flip(frame, 1)
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

@handgestures_control.route('/start_recording')
@login_required
def start_recording():
    global recording, out
    user = current_user.name
    if not recording:
        user_dir = os.path.join('Videos', user)
        os.makedirs(user_dir, exist_ok=True)
        i = 0
        while True:
            video_path = os.path.join(user_dir, f'video{i}.mp4')
            if not os.path.exists(video_path):
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
                recording = True
                break
            i += 1
        return f"Recording started and will be saved as {video_path}"
    return "Already recording"

@handgestures_control.route('/stop_recording')
@login_required
def stop_recording():
    global recording, out
    if recording and out is not None:
        out.release()
        recording = False
        return "Recording stopped and saved"
    return "No active recording to stop"