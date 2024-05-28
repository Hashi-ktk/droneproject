from flask import Blueprint, render_template, Response
from flask_login import login_required, current_user
from cvzone.PIDModule import PID
from djitellopy import Tello
import cv2
from cvzone.PoseModule import PoseDetector
import logging
import os

body_tracking = Blueprint('body_tracking', __name__)

# Initialize the PoseDetector
detector = PoseDetector()

# Camera resolution
hi, wi = 480, 640

# PID controllers for x, y, z axes
xPID = PID([0.22, 0, 0.1], wi // 2)
yPID = PID([0.27, 0, 0.1], hi // 2, axis=1)
zPID = PID([0.00016, 0, 0.000011], 150000, limit=[-20, 15])

# Drone state variables
takeoff = False
land = False
stop_tracking = False
tello = None
recording = False
out = None

# Initialize Tello
def init_tello():
    global tello
    tello = Tello()
    Tello.LOGGER.setLevel(logging.WARNING)
    tello.connect()
    print("Tello battery:", tello.get_battery())

    # Streaming
    tello.streamoff()
    tello.streamon()

# Get frame from stream
def get_frame(hi=hi, wi=wi):
    tello_frame = tello.get_frame_read().frame
    return cv2.resize(tello_frame, (wi, hi))

# Route handler for video stream
@body_tracking.route('/bodytracker_video_feed')
def bodytracker_video_feed():
    global takeoff, land, stop_tracking, tello, recording, out
    stop_tracking = False

    if tello is None:
        init_tello()

    def generate_frames():
        global takeoff, land, stop_tracking, recording, out
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
                    tello.move_up(100)
                    takeoff = True
                except Exception as e:
                    print(f"Error during takeoff: {e}")

            img = get_frame(hi, wi)
            img = detector.findPose(img, draw=True)
            lmList, bboxInfo = detector.findPosition(img, draw=True)

            xVal = 0
            yVal = 0
            zVal = 0

            if bboxInfo:
                cx, cy = bboxInfo['center']
                x, y, w, h = bboxInfo['bbox']
                area = w * h

                xVal = int(xPID.update(cx))
                yVal = int(yPID.update(cy))
                zVal = int(zPID.update(area))

                img = xPID.draw(img, [cx, cy])
                img = yPID.draw(img, [cx, cy])
                cv2.putText(img, str(area), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            tello.send_rc_control(0, -zVal, -yVal, xVal)

            if recording and out is not None:
                out.write(img)
            
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@body_tracking.route('/stop_bodytracking')
def stop_bodytracking():
    global stop_tracking, tello
    stop_tracking = True
    return render_template('profile.html')

@body_tracking.route('/connect_to_bodytracker')
def connect_to_bodytracker():
    return render_template('bodytracker.html')

@body_tracking.route('/capture_image')
@login_required
def capture_image():
    global tello
    user = current_user.name
    img = get_frame(hi, wi)
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

@body_tracking.route('/start_recording')
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
                out = cv2.VideoWriter(video_path, fourcc, 30.0, (wi, hi))
                recording = True
                break
            i += 1
        return f"Recording started and will be saved as {video_path}"
    return "Already recording"

@body_tracking.route('/stop_recording')
def stop_recording():
    global recording, out
    if recording and out is not None:
        out.release()
        recording = False
        return "Recording stopped and saved"
    return "No active recording to stop"
