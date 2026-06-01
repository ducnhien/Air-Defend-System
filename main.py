import cv2
import time

from ultralytics import YOLO

from tracking import PanTiltTracker
from serial_manager import SerialManager

# =================================

PORT = "COM5"

# =================================

model = YOLO("best.pt")

serial_mgr = SerialManager(
    port=PORT,
    baudrate=115200
)

tracker = PanTiltTracker()

# =================================

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# =================================

class_names = {

    0: "Helicopter",
    1: "Jet",
    2: "Rocket"
}

# =================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(
        frame,
        (640, 640)
    )

    distance = serial_mgr.distance
    motion = serial_mgr.motion

    status = "IDLE"

    results = model(
        frame,
        conf=0.5,
        verbose=False
    )

    target_found = False

    if motion == 1:

        for result in results:

            boxes = result.boxes

            for box in boxes:

                conf = float(box.conf)

                cls = int(box.cls)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                label = class_names.get(
                    cls,
                    "Unknown"
                )

                if distance < 5000:
                    target_found = True

                    pan, tilt = tracker.update(
                        cx,
                        cy
                    )

                    serial_mgr.send_servo(
                        pan,
                        tilt
                    )

                    status = "TRACKING"

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    break

    if not target_found:

        if motion == 1:
            status = "SEARCHING"

        else:
            status = "IDLE"

    # Crosshair

    cv2.line(
        frame,
        (320, 0),
        (320, 640),
        (255, 255, 255),
        1
    )

    cv2.line(
        frame,
        (0, 320),
        (640, 320),
        (255, 255, 255),
        1
    )

    # Overlay

    cv2.putText(
        frame,
        f"Distance: {distance} mm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Motion: {motion}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Status: {status}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Pan-Tilt Tracker",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()

serial_mgr.close()

cv2.destroyAllWindows()
