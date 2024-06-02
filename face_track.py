from flask import Blueprint, render_template, Response
import cv2
from djitellopy import Tello
from flask_login import login_required, current_user
import numpy as np
import logging
import time
import os

face_tracking = Blueprint('face_tracking', __name__)

# Width and height of the camera 360, 240
w, h = 360, 240

# PID values for smooth moving
pid = [0.35, 0.35, 0]
pError = 0
pError_y = 0

# Face limit area
faceLimitArea = [8000, 10000]

# Drone state variables
tello = None
takeoff = False
land = False
stop_tracking = False
recording = False
out = None

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

# Detecting frontal faces on the given image
def face_detect(img):
    try:
        frontal_face = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_faces = frontal_face.detectMultiScale(img_gray, 1.2, 8)

        face_list = []
        face_list_area = []

        for (x, y, w, h) in img_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = x + w // 2
            cy = y + h // 2
            face_list.append([cx, cy])
            face_list_area.append(w * h)

        if len(face_list_area) != 0:
            i = face_list_area.index(max(face_list_area))
            return img, [face_list[i], face_list_area[i]]
        else:
            return img, [[0, 0], 0]
    except Exception as e:
        print(f"Error detecting face: {e}")
        return img, [[0, 0], 0]

# Tracking face smoothly with PID
def face_track(tello, face_info, w, h, pid, pError, pError_y):
    try:
        x = face_info[0][0]
        y = face_info[0][1]
        area = face_info[1]
        forw_backw = 0

        error = x - w // 2
        speed = pid[0] * error + pid[1] * (error - pError)
        speed = int(np.clip(speed, -60, 60))

        error_y = h // 2 - y
        speed_y = pid[0] * error_y + pid[1] * (error_y - pError_y)
        speed_y = int(np.clip(speed_y, -60, 60))

        tello.yaw_velocity = speed if x != 0 else 0
        tello.up_down_velocity = speed_y if y != 0 else 0

        if faceLimitArea[0] < area < faceLimitArea[1]:
            forw_backw = 0
        elif area > faceLimitArea[1]:
            forw_backw = -10
        elif area < faceLimitArea[0] and area > 100:
            forw_backw = 10

        tello.send_rc_control(0, forw_backw, tello.up_down_velocity, tello.yaw_velocity)
        return error, error_y
    except Exception as e:
        print(f"Error tracking face: {e}")
        return pError, pError_y

# Route handler for video stream
@face_tracking.route('/facetracker_video_feed')
def facetracker_video_feed():
    global tello, takeoff, pError, pError_y, land, stop_tracking, recording, out
    stop_tracking = False
    tello = init_tello()
    
    def generate_frames():
        global tello, takeoff, pError, pError_y, land, stop_tracking, recording, out
        while True:
            if stop_tracking:
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

            img = get_frame(tello, w, h)
            img, face_info = face_detect(img)
            pError, pError_y = face_track(tello, face_info, w, h, pid, pError, pError_y)

            img = cv2.putText(img, str(tello.get_battery()), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 250), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'pError: {pError}', (0, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'pError_y: {pError_y}', (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'Area: {face_info[1]}', (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1, cv2.LINE_AA)
            
            if recording and out is not None:
                out.write(img)

            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@face_tracking.route('/capture_image')
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

@face_tracking.route('/start_recording')
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

@face_tracking.route('/stop_recording')
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

@face_tracking.route('/stop_facetracking')
def stop_facetracking():
    global stop_tracking
    try:
        stop_tracking = True
        return render_template('profile.html')
    except Exception as e:
        print(f"Error stopping face tracking: {e}")
        return "Failed to stop face tracking"

@face_tracking.route('/connect_to_facetracker')
def connect_to_facetracker():
    return render_template('facetracker.html')
