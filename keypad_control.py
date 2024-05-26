from flask import Blueprint, render_template, Response, request
from djitellopy import Tello
from flask_login import login_required, current_user
import KeyPressModule as kp
from time import sleep
import cv2
import os
import time

keypad_control = Blueprint('keypad_control', __name__)

kp.init()
me = None

# Initialize Tello
def init_tello():
    global me
    if me is None:
        me = Tello()
        me.connect()
        me.streamoff()  # Make sure to turn off the stream before initializing
        me.streamon()

@keypad_control.route('/keypad_video_feed')
def keypad_video_feed():
    global me
    init_tello()  # Initialize Tello drone

    def generate_frames():
        while True:
            frame = me.get_frame_read().frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@keypad_control.route('/control', methods=['POST'])
def control():
    data = request.json
    key = data.get('key')
    speed = 50
    lr, fb, ud, yv = 0, 0, 0, 0
    
    if key == 'LEFT':
        lr = -speed
    elif key == 'RIGHT':
        lr = speed
    elif key == 'UP':
        fb = speed
    elif key == 'DOWN':
        fb = -speed
    elif key == 'w':
        ud = speed
    elif key == 's':
        ud = -speed
    elif key == 'a':
        yv = -speed
    elif key == 'd':
        yv = speed
    elif key == 'f':
        me.flip_right()
    elif key == 'q':
        me.land()
        sleep(3)
    elif key == 'e':
        me.takeoff()
    elif key == 'i':
        capture_image()

    me.send_rc_control(lr, fb, ud, yv)
    
    return '', 204

@keypad_control.route('/connect_to_keypad')
def connect_to_keypad():
    return render_template('keypad.html')

@keypad_control.route('/capture_image')
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

@keypad_control.route('/start_recording')
def start_recording():
    global me
    user = current_user.name
    user_dir = os.path.join('Videos', user)
    os.makedirs(user_dir, exist_ok=True)
    i = 0
    while True:
        video_path = os.path.join(user_dir, f'video{i}.mp4')
        if not os.path.exists(video_path):
            global out
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            break
        i += 1
    return f"Recording started and will be saved as {video_path}"

@keypad_control.route('/stop_recording')
def stop_recording():
    global out
    if out is not None:
        out.release()
        return "Recording stopped and saved"
    return "No active recording to stop"

@keypad_control.route('/stop_keypad_control')
def stop_keypad_control():
    global me
    me.streamoff()
    me.end()
    me = None
    return render_template('profile.html')
