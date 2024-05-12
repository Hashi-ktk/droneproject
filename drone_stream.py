from flask import Blueprint, Response, redirect, url_for
from djitellopy import tello
import cv2

drone_stream = Blueprint('drone_stream', __name__)

me = None  # This will store the Tello object

def generate_frames():
    while True:
        if me:
            img = me.get_frame_read().frame
            img = cv2.resize(img, (360, 240))
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@drone_stream.route('/video_feed')
def video_feed():
    if me:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Drone is not connected"

@drone_stream.route('/connect_to_drone')
def connect_to_drone():
    global me
    if me is None:
        me = tello.Tello()
        me.connect()
        me.streamon()
    return redirect(url_for('main.streaming'))
