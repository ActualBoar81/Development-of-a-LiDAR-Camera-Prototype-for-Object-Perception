import cv2
import numpy as np
from picamera2 import Picamera2
from flask import Flask, Response, jsonify, render_template_string

CHECKERBOARD = (46, 32)
SQUARE_SIZE = 0.025

H = None
latest_frame = None

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

app = Flask(__name__)

HTML = """
<html>
<body>
<h2>Checkerboard Calibration</h2>

<img src="/video" width="700">

<br><br>

<button onclick="fetch('/calibrate')">CALIBRATE & SAVE HOMOGRAPHY</button>

<pre id="status"></pre>

<script>
setInterval(()=>{
 fetch('/status')
 .then(r=>r.json())
 .then(d=>{
 document.getElementById('status').innerText =
 JSON.stringify(d,null,2);
 });
},1000);
</script>

</body>
</html>
"""

# -----------------------------
# PREPROCESS (UPSIDE DOWN FIX)
# -----------------------------
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # UPSIDE DOWN FIX (180° rotation)
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    return frame


# -----------------------------
# HOMOGRAPHY
# -----------------------------
def compute_homography(frame):

    frame = preprocess(frame)   # IMPORTANT: same transform

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD
    )

    if not found:
        return None, False

    corners = corners.reshape(-1, 2)

    cols, rows = CHECKERBOARD

    src = np.array([
        corners[0],
        corners[cols - 1],
        corners[-1],
        corners[-cols]
    ], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [(cols - 1) * SQUARE_SIZE, 0],
        [(cols - 1) * SQUARE_SIZE, (rows - 1) * SQUARE_SIZE],
        [0, (rows - 1) * SQUARE_SIZE]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)

    return H, True


# -----------------------------
# STREAM
# -----------------------------
def generate():

    global latest_frame

    while True:

        frame = picam2.capture_array()

        # ✅ SAME TRANSFORM USED HERE
        frame = preprocess(frame)

        latest_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

        if found:
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, found)
            cv2.putText(frame, "CHECKERBOARD FOUND",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NOT FOUND",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def home():
    return HTML


@app.route("/video")
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/status")
def status():
    return jsonify({"homography_saved": H is not None})


@app.route("/calibrate")
def calibrate():

    global H, latest_frame

    if latest_frame is None:
        return jsonify({"calibrated": False})

    H, found = compute_homography(latest_frame)

    if found:
        np.save("homography.npy", H)

    return jsonify({"calibrated": found})


# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
