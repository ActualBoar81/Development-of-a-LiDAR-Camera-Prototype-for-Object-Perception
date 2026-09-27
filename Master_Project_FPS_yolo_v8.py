import cv2
import time
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MODEL_PATH = "yolov8n.onnx"

TEST_DURATION = 30.0


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print(" YOLOv8n ONNX FPS PERFORMANCE TEST")
print("==============================================")
print()

print(f"[INFO] Resolution    : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
print(f"[INFO] Model         : {MODEL_PATH}")
print(f"[INFO] Test duration : {TEST_DURATION:.0f} seconds")
print()


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print("[INFO] Loading YOLOv8n ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

input_shape = session.get_inputs()[0].shape

print("[INFO] Model loaded.")
print(f"[INFO] Input name    : {input_name}")
print(f"[INFO] Input shape   : {input_shape}")
print("[INFO] Provider      : CPUExecutionProvider")
print()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(frame):

    # YOLOv8 ONNX input
    img = cv2.resize(
        frame,
        (640, 640)
    )

    img = img.astype(
        np.float32
    ) / 255.0

    img = np.transpose(
        img,
        (2, 0, 1)
    )

    img = np.expand_dims(
        img,
        axis=0
    )

    return img


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )
    }
)

picam2.configure(camera_config)

picam2.start()

print("[INFO] Camera started.")
print()


# ============================================================
# WARM-UP
# ============================================================

print("==============================================")
print(" YOLOv8n WARM-UP")
print("==============================================")
print()

print("[INFO] Running 5 warm-up frames...")

for i in range(5):

    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    frame = cv2.flip(
        frame,
        -1
    )

    input_tensor = preprocess(frame)

    _ = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

print("[INFO] Warm-up complete.")
print()


# ============================================================
# FPS MEASUREMENT
# ============================================================

print("==============================================")
print(" STARTING YOLOv8n FPS MEASUREMENT")
print("==============================================")
print()

print(f"Running for {TEST_DURATION:.0f} seconds...")
print("Do not move or restart the Raspberry Pi.")
print()


frame_count = 0

total_pipeline_time = 0.0
total_inference_time = 0.0

fps_samples = []

start_time = time.perf_counter()

last_report_time = start_time


# ============================================================
# MAIN TEST LOOP
# ============================================================

while True:

    current_time = time.perf_counter()

    if current_time - start_time >= TEST_DURATION:
        break


    # --------------------------------------------------------
    # Start complete pipeline timing
    # --------------------------------------------------------

    frame_start = time.perf_counter()


    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    frame = picam2.capture_array()


    # --------------------------------------------------------
    # RGB -> BGR
    # --------------------------------------------------------

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # Same orientation as application
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        -1
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    input_tensor = preprocess(
        frame
    )


    # --------------------------------------------------------
    # ONNX inference timing
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    inference_end = time.perf_counter()

    inference_time = (
        inference_end -
        inference_start
    )


    # --------------------------------------------------------
    # End complete pipeline timing
    # --------------------------------------------------------

    frame_end = time.perf_counter()

    pipeline_time = (
        frame_end -
        frame_start
    )


    # --------------------------------------------------------
    # Store measurements
    # --------------------------------------------------------

    frame_count += 1

    total_pipeline_time += (
        pipeline_time
    )

    total_inference_time += (
        inference_time
    )


    # Instantaneous FPS

    if pipeline_time > 0:

        instant_fps = (
            1.0 /
            pipeline_time
        )

    else:

        instant_fps = 0.0


    fps_samples.append(
        instant_fps
    )


    # --------------------------------------------------------
    # Progress output every 5 seconds
    # --------------------------------------------------------

    if (
        current_time -
        last_report_time
        >= 5.0
    ):

        elapsed = (
            current_time -
            start_time
        )

        average_fps = (
            frame_count /
            elapsed
        )

        print(
            f"Time: {elapsed:5.1f} s | "
            f"Frames: {frame_count:4d} | "
            f"Average FPS: {average_fps:5.2f} | "
            f"Current FPS: {instant_fps:5.2f} | "
            f"YOLO: {inference_time * 1000:6.1f} ms"
        )

        last_report_time = (
            current_time
        )


# ============================================================
# FINAL RESULTS
# ============================================================

end_time = time.perf_counter()

actual_duration = (
    end_time -
    start_time
)

average_fps = (
    frame_count /
    actual_duration
)

average_pipeline_time = (
    total_pipeline_time /
    frame_count
)

average_inference_time = (
    total_inference_time /
    frame_count
)

minimum_fps = min(
    fps_samples
)

maximum_fps = max(
    fps_samples
)


# ============================================================
# RESULTS
# ============================================================

print()
print()

print("==============================================")
print(" YOLOv8n ONNX FPS TEST RESULTS")
print("==============================================")
print()

print(
    f"Model                   : "
    f"{MODEL_PATH}"
)

print(
    f"Resolution              : "
    f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
)

print(
    f"Test duration           : "
    f"{actual_duration:.2f} s"
)

print(
    f"Frames processed        : "
    f"{frame_count}"
)

print()

print(
    f"Average FPS             : "
    f"{average_fps:.2f}"
)

print(
    f"Minimum instantaneous FPS: "
    f"{minimum_fps:.2f}"
)

print(
    f"Maximum instantaneous FPS: "
    f"{maximum_fps:.2f}"
)

print()

print(
    f"Average pipeline time   : "
    f"{average_pipeline_time * 1000:.2f} ms"
)

print(
    f"Average YOLO time       : "
    f"{average_inference_time * 1000:.2f} ms"
)

print()

print("==============================================")
print(" YOLOv8n FPS TEST COMPLETE")
print("==============================================")
print()


# ============================================================
# STOP CAMERA
# ============================================================

picam2.stop()