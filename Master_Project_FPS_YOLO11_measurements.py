import cv2
import time
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MODEL_PATH = "best.pt"

CONF_THRES = 0.35

TEST_DURATION = 30.0   # seconds


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

print()
print("==============================================")
print(" YOLO11 FPS PERFORMANCE TEST")
print("==============================================")
print()

print(
    f"[INFO] Resolution: "
    f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
)

print(
    f"[INFO] Test duration: "
    f"{TEST_DURATION:.0f} seconds"
)

print(
    f"[INFO] Confidence threshold: "
    f"{CONF_THRES}"
)

print()


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


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("[INFO] Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("[INFO] YOLO model loaded.")

print(
    "[INFO] Classes:",
    model.names
)

print()


# ============================================================
# WARM-UP
# ============================================================

print("[INFO] Warming up YOLO model...")

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

    _ = model(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

print("[INFO] Warm-up complete.")
print()


# ============================================================
# FPS TEST
# ============================================================

print("==============================================")
print(" Starting FPS measurement")
print("==============================================")
print()

print(
    f"Running for {TEST_DURATION:.0f} seconds..."
)

print(
    "Do not move or restart the Raspberry Pi."
)

print()


# ------------------------------------------------------------
# Counters
# ------------------------------------------------------------

frame_count = 0

total_pipeline_time = 0.0

total_yolo_time = 0.0

fps_samples = []

start_time = time.perf_counter()

last_report_time = start_time


# ============================================================
# MAIN TEST LOOP
# ============================================================

while True:

    current_time = time.perf_counter()

    # Stop after test duration
    if (
        current_time - start_time
        >= TEST_DURATION
    ):
        break

    # --------------------------------------------------------
    # Start complete frame timer
    # --------------------------------------------------------

    frame_start = time.perf_counter()

    # --------------------------------------------------------
    # Capture frame
    # --------------------------------------------------------

    frame = picam2.capture_array()

    # --------------------------------------------------------
    # Convert RGB -> BGR
    # --------------------------------------------------------

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # Flip image
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        -1
    )

    # --------------------------------------------------------
    # YOLO inference timing
    # --------------------------------------------------------

    yolo_start = time.perf_counter()

    results = model(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

    yolo_end = time.perf_counter()

    yolo_time = (
        yolo_end -
        yolo_start
    )

    # --------------------------------------------------------
    # Process detection count
    # --------------------------------------------------------

    detection_count = 0

    if results:

        if results[0].boxes is not None:

            detection_count = len(
                results[0].boxes
            )

    # --------------------------------------------------------
    # End complete pipeline timer
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

    total_pipeline_time += pipeline_time

    total_yolo_time += yolo_time

    # Instantaneous FPS
    instant_fps = (
        1.0 / pipeline_time
        if pipeline_time > 0
        else 0
    )

    fps_samples.append(
        instant_fps
    )

    # --------------------------------------------------------
    # Periodic terminal output
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
            f"YOLO: {yolo_time*1000:6.1f} ms | "
            f"Objects: {detection_count}"
        )

        last_report_time = current_time


# ============================================================
# FINAL RESULTS
# ============================================================

end_time = time.perf_counter()

actual_duration = (
    end_time -
    start_time
)


# ------------------------------------------------------------
# Overall FPS
# ------------------------------------------------------------

average_fps = (
    frame_count /
    actual_duration
)


# ------------------------------------------------------------
# Average pipeline time
# ------------------------------------------------------------

average_pipeline_time = (
    total_pipeline_time /
    frame_count
)


# ------------------------------------------------------------
# Average YOLO inference time
# ------------------------------------------------------------

average_yolo_time = (
    total_yolo_time /
    frame_count
)


# ------------------------------------------------------------
# Minimum / maximum FPS
# ------------------------------------------------------------

minimum_fps = min(
    fps_samples
)

maximum_fps = max(
    fps_samples
)


# ============================================================
# PRINT FINAL REPORT
# ============================================================

print()
print()
print("==============================================")
print(" FPS TEST RESULTS")
print("==============================================")
print()

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
    f"{average_yolo_time * 1000:.2f} ms"
)

print()

print("==============================================")
print(" FPS TEST COMPLETE")
print("==============================================")
print()


# ============================================================
# RELEASE CAMERA
# ============================================================

picam2.stop()