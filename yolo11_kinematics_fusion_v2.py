import cv2
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
from flask import Flask, Response, jsonify
import serial
import time
import math


# =========================
# CONFIG
# =========================
HOMOGRAPHY_PATH = "homography.npy"

CONF_THRES = 0.35
HOMOGRAPHY_OUTPUT_SCALE_TO_M = 1.0

CHECKERBOARD = (46, 32)
SQUARE_SIZE_M = 0.025

BACKGROUND_DISTANCE_M = None

CAMERA_POS_W_M = np.array([0.0, 0.0, 0.14136])
LIDAR_POS_W_M = np.array([0.0, 0.0, 0.07316])

AIM_X_fraction = 0.65a
AIM_Y_fraction = 0.50

YAW_HOME_DEG = 49.0
PITCH_HOME_DEG = 55.0

YAW_SIGN = -1.0
PITCH_SIGN = 1.0

YAW_SCALE = 1.0
PITCH_SCALE = 1.0

X_BIAS_M = 0.0
Y_BIAS_M = 0.0

# Toggleable booleans
MULTI_OBJECT_MODE = True
USE_VALID_OBJECT_FILTER = True
LATCH_ENABLED = False

# Homography/image correction toggles
FLIP_X = False
FLIP_Y = True

SCAN_MARGIN_M = 0.03

# Multiple latch settings
LATCHED_TARGETS = []
LATCH_DISTANCE_M = 0.05  # 5 cm duplicate threshold

ARDUINO_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200

latest = []
arduino = None

T_WB = None
T_WL0 = None
T_L0W = None


# =========================
# LOAD HOMOGRAPHY
# =========================
H = np.load(HOMOGRAPHY_PATH)


# =========================
# YOLO11 MODEL
# =========================
model = YOLO("best.pt")

print("[INFO] Model loaded")
print("[INFO] Classes:", model.names)

# only needed because your training accidentally used class name "0"
CLASS_REMAP = {
    "0": "box"
}


# =========================
# HOMOGENEOUS TRANSFORMS
# =========================
def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def inv_T(T):
    R = T[:3, :3]
    p = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def setup_transforms():
    if BACKGROUND_DISTANCE_M is None:
        raise RuntimeError("Background distance is not calibrated.")

    R_WB = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    p_WB = np.array([0.0, BACKGROUND_DISTANCE_M, 0.0])
    T_WB_local = make_T(R_WB, p_WB)

    R_WL0 = np.eye(3)
    T_WL0_local = make_T(R_WL0, LIDAR_POS_W_M)

    return T_WB_local, T_WL0_local


def update_transforms_with_background_distance(background_distance_m):
    global BACKGROUND_DISTANCE_M, T_WB, T_WL0, T_L0W

    BACKGROUND_DISTANCE_M = background_distance_m
    T_WB, T_WL0 = setup_transforms()
    T_L0W = inv_T(T_WL0)

    print(f"[System] Background distance updated: {BACKGROUND_DISTANCE_M:.3f} m")


# =========================
# ARDUINO
# =========================
def init_arduino():
    global arduino

    arduino = serial.Serial(ARDUINO_PORT, BAUDRATE, timeout=2)
    time.sleep(2)

    ready = arduino.readline().decode(errors="ignore").strip()
    print("[Arduino]", ready)


def send_angle_and_read_distance(yaw_deg, pitch_deg):
    if arduino is None:
        return None

    yaw_i = round(yaw_deg)
    pitch_i = round(pitch_deg)

    command = f"ANGLE,{yaw_i},{pitch_i}\n"
    print("[Pi -> Arduino]", command.strip())
    arduino.write(command.encode())

    response = arduino.readline().decode(errors="ignore").strip()
    print("[Arduino -> Pi]", response)

    if response.startswith("DIST"):
        distance_raw = float(response.split(",")[1])
        return distance_raw / 100.0

    return None


def calibrate_background_distance():
    print("[System] Calibrating background distance...")
    print("[System] Moving to home position...")

    distance_m = send_angle_and_read_distance(
        YAW_HOME_DEG,
        PITCH_HOME_DEG
    )

    if distance_m is None:
        raise RuntimeError("Could not read background distance from Arduino.")

    update_transforms_with_background_distance(distance_m)
    return distance_m


# =========================
# KINEMATICS
# =========================
def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def background_xy_to_background_point(X, Y):
    X = X * HOMOGRAPHY_OUTPUT_SCALE_TO_M
    Y = Y * HOMOGRAPHY_OUTPUT_SCALE_TO_M

    board_width = (CHECKERBOARD[0] - 1) * SQUARE_SIZE_M
    board_height = (CHECKERBOARD[1] - 1) * SQUARE_SIZE_M

    if FLIP_X:
        X = board_width - X

    if FLIP_Y:
        Y = board_height - Y

    x_B = X - board_width / 2.0
    y_B = Y - board_height / 2.0

    x_B += X_BIAS_M
    y_B += Y_BIAS_M

    return np.array([x_B, y_B, 0.0, 1.0])


def background_point_to_world(p_B):
    if T_WB is None:
        raise RuntimeError("Transforms are not initialized. Run /calibrate first.")
    return T_WB @ p_B


def calculate_yaw_pitch_from_world_point(p_W):
    if T_L0W is None:
        raise RuntimeError("Transforms are not initialized. Run /calibrate first.")

    p_L0 = T_L0W @ p_W

    x = p_L0[0]
    y = p_L0[1]
    z = p_L0[2]

    yaw_rad = math.atan2(x, y)
    pitch_rad = math.atan2(z, math.sqrt(x**2 + y**2))

    return math.degrees(yaw_rad), math.degrees(pitch_rad), p_L0


def map_to_motor_angles(yaw_kin_deg, pitch_kin_deg):
    yaw_motor = YAW_HOME_DEG + YAW_SIGN * YAW_SCALE * yaw_kin_deg
    pitch_motor = PITCH_HOME_DEG + PITCH_SIGN * PITCH_SCALE * pitch_kin_deg

    yaw_motor = clamp(yaw_motor, 0, 180)
    pitch_motor = clamp(pitch_motor, 0, 180)

    return yaw_motor, pitch_motor


def reconstruct_object_point_world(distance_m, yaw_kin_deg, pitch_kin_deg):
    yaw = math.radians(yaw_kin_deg)
    pitch = math.radians(pitch_kin_deg)

    beam_L0 = np.array([
        math.sin(yaw) * math.cos(pitch),
        math.cos(yaw) * math.cos(pitch),
        math.sin(pitch)
    ])

    p_obj_L0 = np.array([
        distance_m * beam_L0[0],
        distance_m * beam_L0[1],
        distance_m * beam_L0[2],
        1.0
    ])

    p_obj_W = T_WL0 @ p_obj_L0
    return p_obj_W.tolist()


# =========================
# CAMERA
# =========================
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()


# =========================
# PIXEL → BACKGROUND
# =========================
def pixel_to_background_xy(cx, cy):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)

    X = float(mapped[0][0][0])
    Y = float(mapped[0][0][1])

    return X, Y


def dist(a, b):
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


# =========================
# MULTIPLE LATCH TARGETS
# =========================
def is_already_latched(obj):
    if obj["p_B_m"] is None:
        return True

    x = obj["p_B_m"][0]
    y = obj["p_B_m"][1]

    for saved in LATCHED_TARGETS:
        sx = saved["p_B_m"][0]
        sy = saved["p_B_m"][1]

        d = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)

        if d < LATCH_DISTANCE_M:
            return True

    return False


def save_latched_target(obj):
    global LATCHED_TARGETS

    if obj["p_B_m"] is None:
        return

    if is_already_latched(obj):
        return

    saved = {
        "class": obj["class"],
        "confidence": obj["confidence"],
        "pixel": obj["pixel"],
        "homography_background_raw": obj["homography_background_raw"],
        "p_B_m": obj["p_B_m"],
        "saved_time": time.time()
    }

    LATCHED_TARGETS.append(saved)
    print("[Latch] Saved target:", saved)


# =========================
# MULTI-OBJECT SCAN HELPERS
# =========================
def is_valid_object_for_scan(obj):
    if obj["p_B_m"] is None:
        return False

    x_B = obj["p_B_m"][0]
    y_B = obj["p_B_m"][1]

    board_width = (CHECKERBOARD[0] - 1) * SQUARE_SIZE_M
    board_height = (CHECKERBOARD[1] - 1) * SQUARE_SIZE_M

    if x_B < -board_width / 2.0 - SCAN_MARGIN_M:
        return False
    if x_B > board_width / 2.0 + SCAN_MARGIN_M:
        return False
    if y_B < -board_height / 2.0 - SCAN_MARGIN_M:
        return False
    if y_B > board_height / 2.0 + SCAN_MARGIN_M:
        return False

    return True


def get_scan_order(objects):
    if USE_VALID_OBJECT_FILTER:
        valid_objects = [
            obj for obj in objects
            if is_valid_object_for_scan(obj)
        ]
    else:
        valid_objects = [
            obj for obj in objects
            if obj["p_B_m"] is not None
        ]

    ordered = sorted(
        valid_objects,
        key=lambda obj: obj["p_B_m"][0]
    )

    return ordered


# =========================
# YOLO-WORLD PROCESSING
# =========================
def process(frame):
    global latest

    results = model(frame, conf=CONF_THRES, verbose=False)
    objects = []

    if len(results) == 0 or results[0].boxes is None:
        latest = []
        return objects, frame

    r = results[0]

    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        label = model.names[int(cls_id)]

        # fix wrongly named class from training
        if label in CLASS_REMAP:
            label = CLASS_REMAP[label]

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cx = int(x1 + AIM_X_fraction * (x2 - x1))
        cy = int(y1 + AIM_Y_fraction * (y2 - y1))

        try:
            bg_X_raw, bg_Y_raw = pixel_to_background_xy(cx, cy)
            p_B = background_xy_to_background_point(bg_X_raw, bg_Y_raw)

        except Exception:
            bg_X_raw, bg_Y_raw = None, None
            p_B = None

        obj = {
            "class": label,
            "confidence": round(conf, 2),
            "pixel": [cx, cy],
            "homography_background_raw": [bg_X_raw, bg_Y_raw],
            "p_B_m": None if p_B is None else p_B.tolist()
        }

        objects.append(obj)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if p_B is not None:
            draw_label = f"{label} {conf:.2f}"
        else:
            draw_label = f"{label} {conf:.2f} - no homography"

        cv2.putText(
            frame,
            draw_label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2
        )

    latest = objects

    if LATCH_ENABLED:
        for obj in objects:
            save_latched_target(obj)

    return objects, frame


# =========================
# FLASK SERVER
# =========================
app = Flask(__name__)


def generate():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, -1)

        _, frame = process(frame)

        _, buffer = cv2.imencode(".jpg", frame)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/")
def home():
    return """
    <h2>YOLO11 + Homography + Scan-Time Kinematics</h2>
    <p><a href="/calibrate">/calibrate</a></p>
    <p><a href="/scan">/scan</a></p>
    <p><a href="/toggle_multi">/toggle_multi</a></p>
    <p><a href="/toggle_filter">/toggle_filter</a></p>
    <p><a href="/toggle_latch">/toggle_latch</a></p>
    <p><a href="/toggle_flip_x">/toggle_flip_x</a></p>
    <p><a href="/toggle_flip_y">/toggle_flip_y</a></p>
    <p><a href="/clear_latch">/clear_latch</a></p>
    <p><a href="/data">/data</a></p>
    <img src="/video" width="800">
    """


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def data():
    return jsonify({
        "background_distance_m": BACKGROUND_DISTANCE_M,
        "multi_object_mode": MULTI_OBJECT_MODE,
        "use_valid_object_filter": USE_VALID_OBJECT_FILTER,
        "latch_enabled": LATCH_ENABLED,
        "latched_count": len(LATCHED_TARGETS),
        "latched_targets": LATCHED_TARGETS,
        "flip_x": FLIP_X,
        "flip_y": FLIP_Y,
        "bias_m": {
            "x": X_BIAS_M,
            "y": Y_BIAS_M
        },
        "model_classes": model.names,
        "latest": latest
    })


@app.route("/clear_latch")
def clear_latch():
    global LATCHED_TARGETS

    LATCHED_TARGETS = []

    return jsonify({
        "status": "cleared",
        "latched_targets": LATCHED_TARGETS
    })


@app.route("/toggle_multi")
def toggle_multi():
    global MULTI_OBJECT_MODE
    MULTI_OBJECT_MODE = not MULTI_OBJECT_MODE
    return jsonify({"multi_object_mode": MULTI_OBJECT_MODE})


@app.route("/toggle_filter")
def toggle_filter():
    global USE_VALID_OBJECT_FILTER
    USE_VALID_OBJECT_FILTER = not USE_VALID_OBJECT_FILTER
    return jsonify({
        "use_valid_object_filter": USE_VALID_OBJECT_FILTER,
        "scan_margin_m": SCAN_MARGIN_M
    })


@app.route("/toggle_latch")
def toggle_latch():
    global LATCH_ENABLED
    LATCH_ENABLED = not LATCH_ENABLED
    return jsonify({
        "latch_enabled": LATCH_ENABLED,
        "latched_count": len(LATCHED_TARGETS),
        "latch_distance_m": LATCH_DISTANCE_M
    })


@app.route("/toggle_flip_x")
def toggle_flip_x():
    global FLIP_X
    FLIP_X = not FLIP_X
    return jsonify({"flip_x": FLIP_X})


@app.route("/toggle_flip_y")
def toggle_flip_y():
    global FLIP_Y
    FLIP_Y = not FLIP_Y
    return jsonify({"flip_y": FLIP_Y})


@app.route("/calibrate")
def calibrate():
    try:
        distance_m = calibrate_background_distance()

        return jsonify({
            "status": "calibrated",
            "background_distance_m": distance_m,
            "home_yaw_deg": YAW_HOME_DEG,
            "home_pitch_deg": PITCH_HOME_DEG
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/scan")
def scan():
    if BACKGROUND_DISTANCE_M is None:
        return jsonify({
            "error": "Background distance is not calibrated. Open /calibrate first."
        })

    if len(LATCHED_TARGETS) > 0:
        objects_to_scan = get_scan_order(LATCHED_TARGETS)
        scan_source = "latched_targets"

    elif MULTI_OBJECT_MODE:
        objects_to_scan = get_scan_order(latest)
        scan_source = "current_multi_object_ordered"

    else:
        if len(latest) == 0:
            return jsonify({"error": "No object detected and no latched target saved."})
        objects_to_scan = [latest[0]]
        scan_source = "current_single_object"

    if len(objects_to_scan) == 0:
        return jsonify({
            "error": "No valid object available for scan.",
            "multi_object_mode": MULTI_OBJECT_MODE,
            "use_valid_object_filter": USE_VALID_OBJECT_FILTER,
            "latch_enabled": LATCH_ENABLED,
            "latched_count": len(LATCHED_TARGETS)
        })

    scan_results = []

    for index, obj in enumerate(objects_to_scan, start=1):
        if obj["p_B_m"] is None:
            scan_results.append({
                "object_index": index,
                "error": "No valid background coordinate"
            })
            continue

        p_B = np.array(obj["p_B_m"])
        p_W = background_point_to_world(p_B)

        yaw_kin, pitch_kin, p_L0 = calculate_yaw_pitch_from_world_point(p_W)
        yaw_motor, pitch_motor = map_to_motor_angles(yaw_kin, pitch_kin)

        distance_m = send_angle_and_read_distance(yaw_motor, pitch_motor)

        if distance_m is None:
            scan_results.append({
                "object_index": index,
                "error": "No valid distance received from Arduino"
            })
            continue

        p_obj_W = reconstruct_object_point_world(
            distance_m,
            yaw_kin,
            pitch_kin
        )

        scan_results.append({
            "object_index": index,
            "object": obj["class"],
            "confidence": obj["confidence"],
            "aim_point_background_m": obj["p_B_m"],
            "aim_point_world_m": p_W.tolist(),
            "aim_point_lidar_home_m": p_L0.tolist(),
            "yaw_kinematic_deg": yaw_kin,
            "pitch_kinematic_deg": pitch_kin,
            "yaw_motor_deg": yaw_motor,
            "pitch_motor_deg": pitch_motor,
            "distance_m": distance_m,
            "reconstructed_object_world_m": p_obj_W
        })

    return jsonify({
        "scan_source": scan_source,
        "scan_mode": "latched_targets" if len(LATCHED_TARGETS) > 0 else (
            "multi_object_ordered" if MULTI_OBJECT_MODE else "single_object_once"
        ),
        "multi_object_mode": MULTI_OBJECT_MODE,
        "use_valid_object_filter": USE_VALID_OBJECT_FILTER,
        "latch_enabled": LATCH_ENABLED,
        "latched_count": len(LATCHED_TARGETS),
        "background_distance_m": BACKGROUND_DISTANCE_M,
        "bias_m": {
            "x": X_BIAS_M,
            "y": Y_BIAS_M
        },
        "results": scan_results
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Starting YOLO-World + Homography + Full Kinematics system...")
    print("Camera position W:", CAMERA_POS_W_M)
    print("LiDAR position W:", LIDAR_POS_W_M)
    print("Bias X/Y:", X_BIAS_M, Y_BIAS_M)
    print("Model classes:", model.names)

    init_arduino()

    print("OPEN:          http://<PI_IP>:5000")
    print("CALIBRATE:     http://<PI_IP>:5000/calibrate")
    print("VIDEO:         http://<PI_IP>:5000/video")
    print("DATA:          http://<PI_IP>:5000/data")
    print("SCAN:          http://<PI_IP>:5000/scan")
    print("CLEAR LATCH:   http://<PI_IP>:5000/clear_latch")
    print("TOGGLE MULTI:  http://<PI_IP>:5000/toggle_multi")
    print("TOGGLE FILTER: http://<PI_IP>:5000/toggle_filter")
    print("TOGGLE LATCH:  http://<PI_IP>:5000/toggle_latch")
    print("TOGGLE FLIP X: http://<PI_IP>:5000/toggle_flip_x")
    print("TOGGLE FLIP Y: http://<PI_IP>:5000/toggle_flip_y")

    app.run(host="0.0.0.0", port=5000, debug=False)
