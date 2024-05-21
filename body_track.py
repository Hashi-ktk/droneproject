from flask import Blueprint, Response, render_template
from cvzone.PIDModule import PID
from djitellopy import Tello
import logging
import cv2
from cvzone.PoseModule import PoseDetector

body_tracking = Blueprint('body_tracking', __name__)

detector = PoseDetector()
me = None  # Initialize me as None initially

def init_tello():
    tello = Tello()
    Tello.LOGGER.setLevel(logging.WARNING)
    tello.connect()
    print("Tello battery:", tello.get_battery())

    # velocity values
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0

    # Streaming
    tello.streamoff()
    tello.streamon()

    return tello

hi, wi = 480, 640
xPID = PID([0.22, 0, 0.1], wi // 2)
yPID = PID([0.27, 0, 0.1], hi // 2, axis=1)
zPID = PID([0.00016, 0, 0.000011], 150000, limit=[-20, 15])

def generate_frames():
    global me  # Accessing the global variable me

    while True:
        img = me.get_frame_read().frame
        img = cv2.resize(img, (640, 480))

        img = detector.findPose(img, draw=True)
        lmList, bboxInfo = detector.findPosition(img, draw=True)

        xVal, yVal, zVal = 0, 0, 0

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

        me.send_rc_control(0, -zVal, -yVal, xVal)

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@body_tracking.route('/bodytracking_video_feed')
def bodytracking_video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@body_tracking.route('/connect_for_bodytracking')
def connect_for_bodytracking():
    me = init_tello()
    return render_template('body_tracking.html')

@body_tracking.route('/stop_bodytracking')
def stop_bodytracking():
    global me  # Accessing the global variable me
    try:
        me.streamoff()
        me.land()
        return render_template('streaming.html')
    except Exception as e:
        print(f"Error during disconnect: {e}")
        return "Failed to stop tracking and land the drone.", 500
