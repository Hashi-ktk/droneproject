from flask import Blueprint, render_template, Response
from cvzone.PIDModule import PID
from djitellopy import Tello
import cv2
from cvzone.PoseModule import PoseDetector
import logging
import time

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

# Initialize Tello
def init_tello():
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

# Get frame from stream
def get_frame(tello, hi=hi, wi=wi):
    tello_frame = tello.get_frame_read().frame
    return cv2.resize(tello_frame, (wi, hi))

# Route handler for video stream
@body_tracking.route('/bodytracker_video_feed')
def bodytracker_video_feed():
    global takeoff, land, stop_tracking
    stop_tracking = False
    tello = init_tello()
    
    def generate_frames():
        global takeoff, land, stop_tracking
        while True:
            if stop_tracking:
                if land:
                    tello.streamoff()
                    tello.land()
                    tello.end()
                    land = False
                break

            if not takeoff:
                try:
                    tello.takeoff()
                    tello.move_up(100)
                    takeoff = True
                    land = True
                    time.sleep(2.2)
                except Exception as e:
                    print(f"Error during takeoff: {e}")

            img = get_frame(tello, hi, wi)
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
            
            ret, jpeg = cv2.imencode('.jpg', img)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@body_tracking.route('/stop_bodytracking')
def stop_bodytracking():
    global stop_tracking
    stop_tracking = True
    return render_template('profile.html')

@body_tracking.route('/connect_to_bodytracker')
def connect_to_bodytracker():
    return render_template('bodytracker.html')
