import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from flask import Flask, Response, jsonify


# ============================================================
# CONFIGURATION
# ============================================================

# Homography file
HOMOGRAPHY_PATH = "homography.npy"

# YOLO confidence threshold
CONF_THRES = 0.35

# Camera resolution
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Checkerboard information
# Used by your physical-coordinate conversion
CHECKERBOARD = (46, 32)
SQUARE_SIZE_M = 0.025

# Reference point location inside YOLO bounding box
#
# X = 0.65 means 65% from left to right
# Y = 0.50 means halfway from top to bottom
AIM_X_fraction = 0.65
AIM_Y_fraction = 0.50

# Optional custom class names
# Example:
# CUSTOM_NAMES = {0: "box", 1: "carton"}
CUSTOM_NAMES = None


# ============================================================
# GLOBAL VARIABLES
# ============================================================

latest = []
LATCHED_TARGET = None


# ============================================================
# LOAD HOMOGRAPHY
# ============================================================

try:
    H = np.load(HOMOGRAPHY_PATH)

    print("[INFO] Homography loaded successfully")
    print("[INFO] H =")
    print(H)

except Exception as e:
    print("[ERROR] Could not load homography:")
    print(e)
    raise


# ============================================================
# LOAD YOLO11 MODEL
# ============================================================

try:
    model = YOLO("best.pt")

    print("[INFO] YOLO model loaded")
    print("[INFO] Classes:", model.names)

except Exception as e:
    print("[ERROR] Could not load YOLO model:")
    print(e)
    raise


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (IMAGE_WIDTH, IMAGE_HEIGHT)
    }
)

picam2.configure(camera_config)

picam2.start()

print(
    f"[INFO] Camera started at "
    f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
)


# ============================================================
# PIXEL -> HOMOGRAPHY COORDINATES
# ============================================================

def pixel_to_background_xy(cx, cy):
    """
    Convert image pixel coordinates (cx, cy)
    into coordinates using the homography matrix.
    """

    point = np.array(
        [[[cx, cy]]],
        dtype=np.float32
    )

    mapped = cv2.perspectiveTransform(
        point,
        H
    )

    bg_x = float(mapped[0][0][0])
    bg_y = float(mapped[0][0][1])

    return bg_x, bg_y


# ============================================================
# HOMOGRAPHY COORDINATES -> PHYSICAL COORDINATES
# ============================================================

def background_xy_to_point(X, Y):
    """
    Convert homography coordinates into the
    project coordinate system.

    The checkerboard dimensions are used to
    define the coordinate origin at the center
    of the board.
    """

    board_w = (
        CHECKERBOARD[0] - 1
    ) * SQUARE_SIZE_M

    board_h = (
        CHECKERBOARD[1] - 1
    ) * SQUARE_SIZE_M

    x_B = X - board_w / 2.0
    y_B = Y - board_h / 2.0

    return [
        x_B,
        y_B,
        0.0
    ]


# ============================================================
# DRAW REFERENCE POINT
# ============================================================

def draw_reference_point(frame, cx, cy):
    """
    Draw a clearly visible reference point.

    Red cross + red circle.
    """

    # Cross
    cv2.drawMarker(
        frame,
        (cx, cy),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=3
    )

    # Circle around reference point
    cv2.circle(
        frame,
        (cx, cy),
        8,
        (0, 0, 255),
        2
    )


# ============================================================
# DRAW TEXT WITH BACKGROUND
# ============================================================

def draw_text_box(
    frame,
    text,
    position,
    font_scale=0.5,
    text_color=(255, 255, 255),
    background_color=(30, 30, 30),
    thickness=1
):
    """
    Draw text with a small background rectangle
    so that it remains readable over the camera image.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness
    )

    cv2.rectangle(
        frame,
        (
            x - 4,
            y - text_height - baseline - 4
        ),
        (
            x + text_width + 4,
            y + 4
        ),
        background_color,
        -1
    )

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS FRAME
# ============================================================

def process(frame):

    global latest

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = model(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

    objects = []

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if not results:
        latest = []
        return objects, frame

    if results[0].boxes is None:
        latest = []
        return objects, frame

    # --------------------------------------------------------
    # Process every detected object
    # --------------------------------------------------------

    r = results[0]

    for box in r.boxes:

        # ====================================================
        # Detection information
        # ====================================================

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        conf = float(
            box.conf[0]
        )

        cls_id = int(
            box.cls[0]
        )

        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        if CUSTOM_NAMES is not None:

            class_name = CUSTOM_NAMES.get(
                cls_id,
                str(cls_id)
            )

        else:

            class_name = model.names[
                cls_id
            ]

        # Preserve your previous "0" -> "box" handling
        if class_name == "0":
            class_name = "box"

        # ----------------------------------------------------
        # Convert bounding box coordinates to integer
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            [x1, y1, x2, y2]
        )

        # ====================================================
        # REFERENCE POINT
        # ====================================================

        # Your existing reference-point definition:
        #
        # X = 65% across bounding box
        # Y = 50% down bounding box

        cx = int(
            x1 +
            AIM_X_fraction *
            (x2 - x1)
        )

        cy = int(
            y1 +
            AIM_Y_fraction *
            (y2 - y1)
        )
        #cx =( x1 + x2)//2
        #cy = ( y1 + y2)//2

        # ====================================================
        # PIXEL -> X,Y
        # ====================================================

        try:

            bg_x, bg_y = pixel_to_background_xy(
                cx,
                cy
            )

            p_B = background_xy_to_point(
                bg_x,
                bg_y
            )

            X = float(p_B[0])
            Y = float(p_B[1])

        except Exception as e:

            print(
                "[WARNING] Homography conversion failed:",
                e
            )

            p_B = None
            X = None
            Y = None

        # ====================================================
        # STORE RESULT
        # ====================================================

        obj = {
            "class": class_name,
            "confidence": round(conf, 3),
            "pixel": [cx, cy],
            "p_B": p_B
        }

        objects.append(obj)

        # ====================================================
        # DRAW BOUNDING BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ====================================================
        # DRAW DETECTION LABEL
        # ====================================================

        label = (
            f"{class_name} "
            f"{conf:.2f}"
        )

        draw_text_box(
            frame,
            label,
            (
                x1,
                max(y1 - 8, 25)
            ),
            font_scale=0.55,
            text_color=(0, 255, 0),
            background_color=(0, 0, 0),
            thickness=2
        )

        # ====================================================
        # DRAW REFERENCE POINT
        # ====================================================

        draw_reference_point(
            frame,
            cx,
            cy
        )

        # ====================================================
        # DRAW PIXEL COORDINATES
        # ====================================================

        pixel_text = (
            f"Pixel: ({cx}, {cy})"
        )

        draw_text_box(
            frame,
            pixel_text,
            (
                min(cx + 15, IMAGE_WIDTH - 180),
                max(cy - 12, 25)
            ),
            font_scale=0.45,
            text_color=(0, 0, 255),
            background_color=(20, 20, 20),
            thickness=1
        )

        # ====================================================
        # DRAW PHYSICAL X,Y
        # ====================================================

        if p_B is not None:

            xy_text = (
                f"X={X:.3f} m  "
                f"Y={Y:.3f} m"
            )

            draw_text_box(
                frame,
                xy_text,
                (
                    min(cx + 15, IMAGE_WIDTH - 210),
                    min(cy + 22, IMAGE_HEIGHT - 120)
                ),
                font_scale=0.48,
                text_color=(255, 0, 0),
                background_color=(20, 20, 20),
                thickness=2
            )

        else:

            draw_text_box(
                frame,
                "X,Y unavailable",
                (
                    min(cx + 15, IMAGE_WIDTH - 180),
                    min(cy + 22, IMAGE_HEIGHT - 120)
                ),
                font_scale=0.48,
                text_color=(0, 0, 255),
                background_color=(20, 20, 20),
                thickness=2
            )

    # ========================================================
    # REPORT INFORMATION PANEL
    # ========================================================

    if len(objects) > 0:

        # ----------------------------------------------------
        # Use first detected object for the report panel
        # ----------------------------------------------------

        obj = objects[0]

        panel_height = 110

        panel_y = IMAGE_HEIGHT - panel_height

        # ----------------------------------------------------
        # Create dark overlay
        # ----------------------------------------------------

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, panel_y),
            (IMAGE_WIDTH, IMAGE_HEIGHT),
            (25, 25, 25),
            -1
        )

        # Transparency
        frame = cv2.addWeighted(
            overlay,
            0.82,
            frame,
            0.18,
            0
        )

        # ----------------------------------------------------
        # Panel title
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "OBJECT LOCALIZATION",
            (15, panel_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Object + confidence
        # ----------------------------------------------------

        cv2.putText(
            frame,
            (
                f"Object: {obj['class']}    "
                f"Confidence: {obj['confidence']:.2f}"
            ),
            (15, panel_y + 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Reference point
        # ----------------------------------------------------

        cx, cy = obj["pixel"]

        cv2.putText(
            frame,
            (
                f"Reference point: "
                f"({cx}, {cy}) px"
            ),
            (15, panel_y + 67),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Physical coordinates
        # ----------------------------------------------------

        if obj["p_B"] is not None:

            X = obj["p_B"][0]
            Y = obj["p_B"][1]

            cv2.putText(
                frame,
                (
                    f"Workspace position: "
                    f"X = {X:.3f} m    "
                    f"Y = {Y:.3f} m"
                ),
                (15, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        else:

            cv2.putText(
                frame,
                "Workspace position: unavailable",
                (15, panel_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    # ========================================================
    # SAVE LATEST DATA
    # ========================================================

    latest = objects

    return objects, frame


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# VIDEO GENERATOR
# ============================================================

def generate():

    while True:

        # ----------------------------------------------------
        # Capture frame
        # ----------------------------------------------------

        frame = picam2.capture_array()

        # ----------------------------------------------------
        # Convert RGB -> BGR for OpenCV
        # ----------------------------------------------------

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        # ----------------------------------------------------
        # Preserve your existing camera orientation
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            -1
        )

        # ----------------------------------------------------
        # YOLO + localization
        # ----------------------------------------------------

        _, frame = process(
            frame
        )

        # ----------------------------------------------------
        # Encode JPEG
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        # ----------------------------------------------------
        # Flask MJPEG stream
        # ----------------------------------------------------

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            +
            frame_bytes
            +
            b"\r\n"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return """
    <!DOCTYPE html>
    <html>

    <head>

        <title>
            YOLO11 Object Localization
        </title>

        <style>

            body {
                font-family: Arial, sans-serif;
                background: #f2f2f2;
                text-align: center;
                margin: 20px;
            }

            h1 {
                margin-bottom: 15px;
            }

            img {
                width: 640px;
                max-width: 95%;
                border: 2px solid #333;
            }

            .info {
                margin-top: 15px;
                font-size: 16px;
            }

        </style>

    </head>

    <body>

        <h1>
            YOLO11 Object Detection and Localization
        </h1>

        <img src="/video">

        <div class="info">
            Camera: 640 × 480
            <br>
            Detection → Reference Point → X,Y
        </div>

    </body>

    </html>
    """


# ============================================================
# VIDEO ROUTE
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# DATA ROUTE
# ============================================================

@app.route("/data")
def data():

    return jsonify(
        {
            "latest": latest
        }
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print(" YOLO11 OBJECT DETECTION + LOCALIZATION")
    print("==============================================")
    print()
    print(
        f"Camera resolution: "
        f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
    )
    print(
        f"Confidence threshold: "
        f"{CONF_THRES}"
    )
    print(
        f"Reference point fractions: "
        f"X={AIM_X_fraction}, "
        f"Y={AIM_Y_fraction}"
    )
    print()
    print("Starting Flask server...")
    print()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )