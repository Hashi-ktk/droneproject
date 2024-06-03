import cv2
import imutils
import datetime
from djitellopy import Tello
from flask import Blueprint, render_template, Response
from flask_login import login_required, current_user
import os
import pygame

pygame.init()
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("Alarms/security-alarm.mp3")

gun_detecting = Blueprint('gun_detecting', __name__)

# Load the cascade classifier for gun detection
gun_cascade = cv2.CascadeClassifier('Resources/cascade.xml')

# Global variables
me = None
stop_tracking = False
recording = False
out = None

# Initialize Tello
def init_tello():
    global me
    if me is None:
        me = Tello()
        me.connect()
        me.streamoff()
        me.streamon()
    return me

@gun_detecting.route('/gundetecter_video_feed')
def gundetecter_video_feed():
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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gun = gun_cascade.detectMultiScale(gray, 1.3, 20, minSize=(100, 100))

            if len(gun) > 0:
                if not alarm_playing:
                    alarm_sound.play(-1)
                    alarm_playing = True
            else:
                if alarm_playing:
                    alarm_sound.stop()
                    alarm_playing = False

            for (x, y, w, h) in gun:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

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

@gun_detecting.route('/capture_image')
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

@gun_detecting.route('/start_recording')
@login_required
def start_recording():
    global me, out, recording
    if recording:
        return "Already recording"

    user = current_user.name
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

@gun_detecting.route('/stop_recording')
@login_required
def stop_recording():
    global out, recording
    if recording and out is not None:
        out.release()
        out = None
        recording = False
        return "Recording stopped and saved"
    return "No active recording to stop"

@gun_detecting.route('/connect_to_gundetecter')
def connect_to_gundetecter():
    return render_template('gundetecter.html')

@gun_detecting.route('/stop_gundetecting')
def stop_gundetecting():
    global stop_tracking
    stop_tracking = True
    return "Gun detection stopped"
