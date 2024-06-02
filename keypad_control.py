from flask import Blueprint, render_template, Response, request
from djitellopy import Tello
from flask_login import login_required, current_user
import KeyPressModule as kp
import cv2
import os
from time import sleep

keypad_control = Blueprint('keypad_control', __name__)

kp.init()
me = None
out = None

# Initialize Tello
def init_tello():
    global me
    if me is None:
        me = Tello()
        me.connect()
        me.streamoff()
        me.streamon()
        print("Tello initialized and connected")

@keypad_control.route('/keypad_video_feed')
def keypad_video_feed():
    global me
    init_tello()  # Initialize Tello drone

    def generate_frames():
        while True:
            if me:
                frame = me.get_frame_read().frame
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@keypad_control.route('/control', methods=['POST'])
def control():
    global me
    data = request.json
    key = data.get('key')
    speed = 50
    lr, fb, ud, yv = 0, 0, 0, 0
    
    if me:
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
@login_required
def capture_image():
    global me
    user = current_user.name
    if me:
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
    return "Failed to capture image"

@keypad_control.route('/start_recording')
@login_required
def start_recording():
    global me, out
    user = current_user.name
    if not out:
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
    return "Already recording"

@keypad_control.route('/stop_recording')
@login_required
def stop_recording():
    global out
    if out:
        out.release()
        out = None
        return "Recording stopped and saved"
    return "No active recording to stop"

@keypad_control.route('/stop_keypad_control')
def stop_keypad_control():
    global me, out
    if me:
        me.streamoff()
        me.end()
        me = None
    if out:
        out.release()
        out = None
    return render_template('profile.html')
 