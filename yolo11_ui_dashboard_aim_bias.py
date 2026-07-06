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

AIM_X_fraction = 0.65
AIM_Y_fraction = 0.50

YAW_HOME_DEG = 49.0
PITCH_HOME_DEG = 55.0

YAW_SIGN = -1.0
PITCH_SIGN = 1.0

YAW_SCALE = 1.0
PITCH_SCALE = 1.0

X_BIAS_M = 0.0
Y_BIAS_M = 0.0

# Aim bias shifts the target point on the background plane.
# If LiDAR hits top-left of the object, use +X and -Y to aim bottom-right.
AIM_BIAS_X_M = 0.01
AIM_BIAS_Y_M = -0.01
AIM_BIAS_STEP_M = 0.005

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
LAST_SCAN_RESULT = None
LAST_CALIBRATION_RESULT = None
LAST_OPERATION_RESULT = None
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

    # General bias + aim bias correction on the background plane
    x_B += X_BIAS_M + AIM_BIAS_X_M
    y_B += Y_BIAS_M + AIM_BIAS_Y_M

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

   if LATCH_ENABLED and len(objects) > 0:
    if MULTI_OBJECT_MODE:
        for obj in objects:
            save_latched_target(obj)
    else:
        save_latched_target(objects[0])

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
<!doctype html>
<html>
<head>
    <title>YOLO11 LiDAR Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; background: #111827; color: #e5e7eb; }
        .page { display: grid; grid-template-columns: 820px 1fr; gap: 18px; padding: 18px; }
        .card { background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 14px; }
        h2, h3 { margin-top: 0; }
        img { width: 800px; border-radius: 8px; border: 1px solid #374151; }
        button { border: 0; border-radius: 8px; padding: 10px 12px; margin: 4px; color: white; cursor: pointer; font-weight: 600; }
        button.action { background: #2563eb; }
        button.on { background: #16a34a; }
        button.off { background: #dc2626; }
        button.clear { background: #7c2d12; }
        .grid { display: grid; grid-template-columns: repeat(2, minmax(130px, 1fr)); gap: 6px; }
        .status-line { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #374151; }
        .small { color: #9ca3af; font-size: 13px; }
        table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
        th, td { border-bottom: 1px solid #374151; padding: 6px; text-align: left; }
        th { color: #93c5fd; }
        pre { white-space: pre-wrap; word-break: break-word; background: #0b1220; padding: 10px; border-radius: 8px; max-height: 220px; overflow: auto; }
    </style>
</head>
<body>
    <div class="page">
        <div class="card">
            <h2>YOLO11 + LiDAR Kinematics</h2>
            <img src="/video">
        </div>

        <div class="card">
            <h3>Operations</h3>
            <button class="action" onclick="callRoute('/calibrate')">Calibrate Background</button>
            <button class="action" onclick="callRoute('/scan')">Scan</button>
            <button class="clear" onclick="callRoute('/clear_latch')">Clear Latch</button>

            <h3>Toggles</h3>
            <div class="grid">
                <button id="btn_multi" onclick="callRoute('/toggle_multi')">Multi Object</button>
                <button id="btn_filter" onclick="callRoute('/toggle_filter')">Valid Filter</button>
                <button id="btn_latch" onclick="callRoute('/toggle_latch')">Latch</button>
                <button id="btn_flipx" onclick="callRoute('/toggle_flip_x')">Flip X</button>
                <button id="btn_flipy" onclick="callRoute('/toggle_flip_y')">Flip Y</button>
            </div>

            <h3>Aim Bias</h3>
            <div class="grid">
                <button class="action" onclick="callRoute('/aim_left')">Aim Left</button>
                <button class="action" onclick="callRoute('/aim_right')">Aim Right</button>
                <button class="action" onclick="callRoute('/aim_up')">Aim Up</button>
                <button class="action" onclick="callRoute('/aim_down')">Aim Down</button>
                <button class="clear" onclick="callRoute('/reset_aim_bias')">Reset Aim Bias</button>
            </div>

            <h3>System Status</h3>
            <div id="status"></div>

            <h3>Last Operation Result</h3>
            <pre id="operation">No operation yet.</pre>
        </div>
    </div>

    <div class="card" style="margin: 0 18px 18px 18px;">
        <h3>Scan Results</h3>
        <div class="small">Object name, distance, yaw, pitch, motor angles, and reconstructed world coordinate.</div>
        <div id="scan_table">No scan result yet.</div>
    </div>

<script>
function fmt(v, digits=3) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "number") return v.toFixed(digits);
    return v;
}

function setButton(id, state, label) {
    const b = document.getElementById(id);
    if (!b) return;
    b.className = state ? "on" : "off";
    b.innerText = label + ": " + (state ? "ON" : "OFF");
}

function renderStatus(d) {
    setButton("btn_multi", d.multi_object_mode, "Multi Object");
    setButton("btn_filter", d.use_valid_object_filter, "Valid Filter");
    setButton("btn_latch", d.latch_enabled, "Latch");
    setButton("btn_flipx", d.flip_x, "Flip X");
    setButton("btn_flipy", d.flip_y, "Flip Y");

    document.getElementById("status").innerHTML = `
        <div class="status-line"><span>Background distance</span><b>${fmt(d.background_distance_m)} m</b></div>
        <div class="status-line"><span>Latest detections</span><b>${d.latest ? d.latest.length : 0}</b></div>
        <div class="status-line"><span>Latched targets</span><b>${d.latched_count}</b></div>
        <div class="status-line"><span>X bias</span><b>${fmt(d.bias_m?.x)} m</b></div>
        <div class="status-line"><span>Y bias</span><b>${fmt(d.bias_m?.y)} m</b></div>
        <div class="status-line"><span>Aim bias X</span><b>${fmt(d.aim_bias_m?.x)} m</b></div>
        <div class="status-line"><span>Aim bias Y</span><b>${fmt(d.aim_bias_m?.y)} m</b></div>
        <div class="status-line"><span>Aim bias step</span><b>${fmt(d.aim_bias_step_m)} m</b></div>
    `;
}

function renderScan(data) {
    if (!data || !data.results || data.results.length === 0) {
        document.getElementById("scan_table").innerHTML = "No scan result.";
        return;
    }
    let rows = data.results.map(r => {
        const world = r.reconstructed_object_world_m || [];
        return `<tr>
            <td>${r.object_index ?? "-"}</td>
            <td>${r.object ?? "-"}</td>
            <td>${fmt(r.confidence, 2)}</td>
            <td>${fmt(r.distance_m)} m</td>
            <td>${fmt(r.yaw_kinematic_deg, 2)}</td>
            <td>${fmt(r.pitch_kinematic_deg, 2)}</td>
            <td>${fmt(r.yaw_motor_deg, 2)}</td>
            <td>${fmt(r.pitch_motor_deg, 2)}</td>
            <td>[${world.map(x => fmt(x)).join(", ")}]</td>
        </tr>`;
    }).join("");

    document.getElementById("scan_table").innerHTML = `
        <div class="small">Source: ${data.scan_source || "-"} | Mode: ${data.scan_mode || "-"}</div>
        <table>
            <tr>
                <th>#</th><th>Object</th><th>Conf.</th><th>Distance</th>
                <th>Yaw kin.</th><th>Pitch kin.</th><th>Yaw motor</th><th>Pitch motor</th><th>XYZ world</th>
            </tr>
            ${rows}
        </table>`;
}

async function refresh() {
    try {
        const r = await fetch('/data');
        const d = await r.json();
        renderStatus(d);
        if (d.last_scan_result) renderScan(d.last_scan_result);
    } catch (e) {
        console.log(e);
    }
}

async function callRoute(route) {
    const op = document.getElementById("operation");
    op.innerText = "Running " + route + " ...";
    try {
        const r = await fetch(route);
        const d = await r.json();
        op.innerText = JSON.stringify(d, null, 2);
        if (route === '/scan') renderScan(d);
        await refresh();
    } catch (e) {
        op.innerText = "Error: " + e;
    }
}

setInterval(refresh, 1000);
refresh();
</script>
</body>
</html>
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
        "aim_bias_m": {
            "x": AIM_BIAS_X_M,
            "y": AIM_BIAS_Y_M
        },
        "aim_bias_step_m": AIM_BIAS_STEP_M,
        "model_classes": model.names,
        "latest": latest,
        "last_scan_result": LAST_SCAN_RESULT,
        "last_calibration_result": LAST_CALIBRATION_RESULT,
        "last_operation_result": LAST_OPERATION_RESULT
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


@app.route("/aim_right")
def aim_right():
    global AIM_BIAS_X_M
    AIM_BIAS_X_M += AIM_BIAS_STEP_M
    return jsonify({"aim_bias_x_m": AIM_BIAS_X_M, "aim_bias_y_m": AIM_BIAS_Y_M})


@app.route("/aim_left")
def aim_left():
    global AIM_BIAS_X_M
    AIM_BIAS_X_M -= AIM_BIAS_STEP_M
    return jsonify({"aim_bias_x_m": AIM_BIAS_X_M, "aim_bias_y_m": AIM_BIAS_Y_M})


@app.route("/aim_up")
def aim_up():
    global AIM_BIAS_Y_M
    AIM_BIAS_Y_M += AIM_BIAS_STEP_M
    return jsonify({"aim_bias_x_m": AIM_BIAS_X_M, "aim_bias_y_m": AIM_BIAS_Y_M})


@app.route("/aim_down")
def aim_down():
    global AIM_BIAS_Y_M
    AIM_BIAS_Y_M -= AIM_BIAS_STEP_M
    return jsonify({"aim_bias_x_m": AIM_BIAS_X_M, "aim_bias_y_m": AIM_BIAS_Y_M})


@app.route("/reset_aim_bias")
def reset_aim_bias():
    global AIM_BIAS_X_M, AIM_BIAS_Y_M
    AIM_BIAS_X_M = 0.0
    AIM_BIAS_Y_M = 0.0
    return jsonify({"aim_bias_x_m": AIM_BIAS_X_M, "aim_bias_y_m": AIM_BIAS_Y_M})


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
    global LAST_CALIBRATION_RESULT, LAST_OPERATION_RESULT

    try:
        distance_m = calibrate_background_distance()

        result = {
            "status": "calibrated",
            "background_distance_m": distance_m,
            "home_yaw_deg": YAW_HOME_DEG,
            "home_pitch_deg": PITCH_HOME_DEG
        }

    except Exception as e:
        result = {"error": str(e)}

    LAST_CALIBRATION_RESULT = result
    LAST_OPERATION_RESULT = result
    return jsonify(result)

@app.route("/scan")
def scan():
    global LAST_SCAN_RESULT, LAST_OPERATION_RESULT

    if BACKGROUND_DISTANCE_M is None:
        result = {
            "error": "Background distance is not calibrated. Open /calibrate first."
        }
        LAST_SCAN_RESULT = result
        LAST_OPERATION_RESULT = result
        return jsonify(result)

    if len(LATCHED_TARGETS) > 0:
        objects_to_scan = get_scan_order(LATCHED_TARGETS)
        scan_source = "latched_targets"

    elif MULTI_OBJECT_MODE:
        objects_to_scan = get_scan_order(latest)
        scan_source = "current_multi_object_ordered"

    else:
        if len(latest) == 0:
            result = {"error": "No object detected and no latched target saved."}
            LAST_SCAN_RESULT = result
            LAST_OPERATION_RESULT = result
            return jsonify(result)
        objects_to_scan = [latest[0]]
        scan_source = "current_single_object"

    if len(objects_to_scan) == 0:
        result = {
            "error": "No valid object available for scan.",
            "multi_object_mode": MULTI_OBJECT_MODE,
            "use_valid_object_filter": USE_VALID_OBJECT_FILTER,
            "latch_enabled": LATCH_ENABLED,
            "latched_count": len(LATCHED_TARGETS)
        }
        LAST_SCAN_RESULT = result
        LAST_OPERATION_RESULT = result
        return jsonify(result)

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

    result = {
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
        "aim_bias_m": {
            "x": AIM_BIAS_X_M,
            "y": AIM_BIAS_Y_M
        },
        "results": scan_results
    }

    LAST_SCAN_RESULT = result
    LAST_OPERATION_RESULT = result
    return jsonify(result)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    print("Starting YOLO-World + Homography + Full Kinematics system...")
    print("Camera position W:", CAMERA_POS_W_M)
    print("LiDAR position W:", LIDAR_POS_W_M)
    print("Bias X/Y:", X_BIAS_M, Y_BIAS_M)
    print("Aim bias X/Y:", AIM_BIAS_X_M, AIM_BIAS_Y_M)
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
