import cv2
from djitellopy import Tello
from flask import Blueprint, render_template, Response, request
from flask_login import login_required, current_user
import os
import pygame
import datetime
import imutils

pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("Alarms/fire_alarm.mp3")

fire_detecting = Blueprint('fire_detecting', __name__)

# Load the cascade classifier for fire detection
fire_cascade = cv2.CascadeClassifier('Resources/fire_detection_cascade_model.xml')

me = None
stop_tracking = False

# Initialize Tello
def init_tello():
    global me
    me = Tello()
    me.connect()
    me.streamoff()
    me.streamon()
    return me

@fire_detecting.route('/fire_detecting_video_feed')
def fire_detecting_video_feed():
    global me, stop_tracking
    me = init_tello()
    me.takeoff()
    me.move_up(80)
    stop_tracking = False

    def generate_frames():
        alarm_playing = False

        while not stop_tracking:
            frame = me.get_frame_read().frame
            if frame is None:
                continue

            frame = imutils.resize(frame, width=500)

            # Convert frame to grayscale for cascade classifier
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect fire using cascade classifier
            fires = fire_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            fire_detected = False
            for (x, y, w, h) in fires:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                fire_detected = True

            if fire_detected:
                if not alarm_playing:
                    alarm_sound.play(-1)
                    alarm_playing = True
            else:
                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False

            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1)
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if alarm_playing:
            alarm_sound.stop()
        me.land()
        me.end()
        me = None

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@fire_detecting.route('/capture_image')
@login_required
def capture_image():
    global me
    user = current_user.name
    img = me.get_frame_read().frame
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

@fire_detecting.route('/start_recording')
@login_required
def start_recording():
    global me, out
    user = current_user.name
    user_dir = os.path.join('Videos', user)
    os.makedirs(user_dir, exist_ok=True)
    i = 0
    while True:
        video_path = os.path.join(user_dir, f'video{i}.mp4')
        if not os.path.exists(video_path):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            break
        i += 1
    return f"Recording started and will be saved as {video_path}"

@fire_detecting.route('/stop_recording')
@login_required
def stop_recording():
    global out
    if out is not None:
        out.release()
        out = None
        return "Recording stopped and saved"
    return "No active recording to stop"

@fire_detecting.route('/connect_to_fire_detecting')
def connect_to_fire_detecting():
    return render_template('firedetecter.html')

@fire_detecting.route('/stop_fire_detecting')
def stop_fire_detecting():
    global stop_tracking
    stop_tracking = True
    return "Fire detection stopped"
