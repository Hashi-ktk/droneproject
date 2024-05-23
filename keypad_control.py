from flask import Blueprint, render_template, Response, request
from djitellopy import Tello
from flask_login import login_required, current_user
import KeyPressModule as kp
from time import sleep
import cv2
import os

keypad_control = Blueprint('keypad_control', __name__)

kp.init()
me = None

# Initialize Tello
def init_tello():
    global me
    me = Tello()
    me.connect()
    me.streamoff()  # Make sure to turn off the stream before initializing
    me.streamon()

@keypad_control.route('/keypad_video_feed')
def keypad_video_feed():
    global me
    me = init_tello()  # Initialize Tello drone

    def getKeyboardInput():
        lr, fb, ud, yv = 0, 0, 0, 0
        speed = 50
        if kp.getKey("LEFT"):
            lr = -speed
        elif kp.getKey("RIGHT"):
            lr = speed
        if kp.getKey("UP"):
            fb = speed
        elif kp.getKey("DOWN"):
            fb = -speed
        if kp.getKey("w"):
            ud = speed
        elif kp.getKey("s"):
            ud = -speed
        if kp.getKey("a"):
            yv = -speed
        elif kp.getKey("d"):
            yv = speed
        if kp.getKey("f"):
            me.flip_right()
        if kp.getKey("q"):
            me.land()
            sleep(3)
        if kp.getKey("e"):
            me.takeoff()
        return [lr, fb, ud, yv]

    def generate_frames():
        while True:
            vals = getKeyboardInput()
            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            frame = me.get_frame_read().frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@keypad_control.route('/connect_to_keypad')
def connect_to_keypad():
    return render_template('keypad.html')

@keypad_control.route('/capture_image')
def capture_image():
    global me
    user= current_user.name
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
