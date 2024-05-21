from flask import Blueprint, Response, redirect, url_for
from flask_login import current_user, login_required
from djitellopy import Tello
import cv2
import os

drone_stream = Blueprint('drone_stream', __name__)
me = None
recording = None
out = None

# Function to generate frames for streaming
def generate_frames():
    global me
    while True:
        if me:
            img = me.get_frame_read().frame
            img = cv2.resize(img, (360, 240))
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if recording:
                out.write(img)

# Function to start recording video
@drone_stream.route('/start_recording')
@login_required
def start_recording():
    global recording, out
    # Directory paths for storing recordings and images
    RECORDINGS_DIR = os.path.join("Recordings", "Hashir")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    

    recording = True
    out = cv2.VideoWriter(os.path.join(RECORDINGS_DIR, f"video_{len(os.listdir(RECORDINGS_DIR)) + 1}.avi"),
                          cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (640, 480))
    return "Recording started"

# Function to stop recording video
@drone_stream.route('/stop_recording')
@login_required
def stop_recording():
    global recording, out
    recording = False
    if out:
        out.release()
    return "Recording stopped"

# Function to capture image
@drone_stream.route('/capture_image')
@login_required
def capture_image():
    global me
    # Directory paths for storing recordings and images
    IMAGES_DIR = os.path.join("Images", "Hashir")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    if me:
        img = me.get_frame_read().frame
        filename = os.path.join(IMAGES_DIR, f"image_{len(os.listdir(IMAGES_DIR)) + 1}.jpg")
        cv2.imwrite(filename, img)
        return f"Image captured and saved as {filename}"
    else:
        return "Drone is not connected"

# Function to stream video feed
@drone_stream.route('/video_feed')
@login_required
def video_feed():
    global me
    if me:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Drone is not connected"

# Function to connect to the drone
@drone_stream.route('/connect_to_drone')
@login_required
def connect_to_drone():
    global me
    if me is None:
        me = Tello()
        me.connect()
        me.streamon()
    return redirect(url_for('main.streaming'))

# Function to disconnect from the drone
@drone_stream.route("/disconnect_drone")
@login_required
def disconnect_drone():
    global me
    if me:
        me.streamoff()
        me.land()
    return redirect(url_for('main.profile'))
