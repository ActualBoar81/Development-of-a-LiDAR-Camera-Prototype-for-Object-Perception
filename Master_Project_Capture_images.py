from flask import Flask, render_template, Response, request, send_from_directory
from picamera2 import Picamera2
import cv2
import os
import time

app = Flask(__name__)

save_path = "calib_images"
os.makedirs(save_path, exist_ok=True)

# Initialize Pi Camera
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480)}
)

picam2.configure(config)

# OPTIONAL: lock focus for calibration
picam2.set_controls({
    "AfMode": 0,
    "LensPosition": 1.2
})

picam2.start()

image_count = 0


def generate_frames():
    while True:
        frame = picam2.capture_array()

        # Convert RGB → BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        ret, buffer = cv2.imencode('.jpg', frame)

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture', methods=['POST'])
def capture():
    global image_count

    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    filename = f"{save_path}/img_{image_count}.jpg"

    cv2.imwrite(filename, frame)

    print("Saved:", filename)

    image_count += 1

    return ("Image Captured!", 200)

@app.route('/gallery')
def gallery():
    images = os.listdir("calib_images_2")
    images = [img for img in images if img.endswith(".jpg")]
    images.sort()
    return render_template("gallery.html", images=images)


@app.route('/calib_images_2/<filename>')
def get_image(filename):
    return send_from_directory("calib_images", filename)

#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port=5000)
import os

if __name__ == '__main__':

    ip = os.popen("hostname -I").read().split()[0]

    print("\n===================================")
    print(" Raspberry Pi Camera Web Interface")
    print("===================================\n")

    print(f"Camera Stream:")
    print(f"http://{ip}:5000\n")

    print(f"Calibration Gallery:")
    print(f"http://{ip}:5000/gallery\n")

    print("===================================\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
