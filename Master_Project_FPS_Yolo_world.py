import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MODEL_PATH = "yolov8s-world.pt"

CONF_THRES = 0.35

TEST_DURATION = 30.0

# YOLO-World prompt classes
PROMPT_CLASSES = [
    "person",
    "chair",
    "table",
    "sofa",
    "bottle",
    "laptop",
    "box",
    "cardboard box",
    "carton",
    "package",
    "parcel",
    "crate"
]


# ============================================================
# START
# ============================================================

print()
print("==============================================")
print(" YOLO-WORLD FPS PERFORMANCE TEST")
print("==============================================")
print()

print(f"[INFO] Resolution       : {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
print(f"[INFO] Model            : {MODEL_PATH}")
print(f"[INFO] Test duration    : {TEST_DURATION:.0f} seconds")
print(f"[INFO] Confidence       : {CONF_THRES}")
print()

print("[INFO] YOLO-World classes:")
for i, name in enumerate(PROMPT_CLASSES):
    print(f"       {i}: {name}")

print()


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
# LOAD YOLO-WORLD
# ============================================================

print("[INFO] Loading YOLO-World model...")

model = YOLO(MODEL_PATH)

print("[INFO] YOLO-World model loaded.")

# Set open-vocabulary classes BEFORE measurement
print("[INFO] Setting YOLO-World classes...")

model.set_classes(PROMPT_CLASSES)

print("[INFO] YOLO-World classes configured.")
print()


# ============================================================
# WARM-UP
# ============================================================

print("==============================================")
print(" YOLO-WORLD WARM-UP")
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

    _ = model(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

print("[INFO] Warm-up complete.")
print()


# ============================================================
# FPS MEASUREMENT
# ============================================================

print("==============================================")
print(" STARTING YOLO-WORLD FPS MEASUREMENT")
print("==============================================")
print()

print(f"Running for {TEST_DURATION:.0f} seconds...")
print("Do not move or restart the Raspberry Pi.")
print()


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

    elapsed = current_time - start_time

    if elapsed >= TEST_DURATION:
        break


    # --------------------------------------------------------
    # Start total pipeline timing
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
    # Match orientation used by your application
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        -1
    )


    # --------------------------------------------------------
    # YOLO-WORLD inference timing
    # --------------------------------------------------------

    yolo_start = time.perf_counter()

    results = model(
        frame,
        conf=CONF_THRES,
        verbose=False
    )

    yolo_end = time.perf_counter()

    yolo_time = yolo_end - yolo_start


    # --------------------------------------------------------
    # Count detections
    # --------------------------------------------------------

    detection_count = 0

    if results:

        if results[0].boxes is not None:

            detection_count = len(
                results[0].boxes
            )


    # --------------------------------------------------------
    # End total pipeline timing
    # --------------------------------------------------------

    frame_end = time.perf_counter()

    pipeline_time = frame_end - frame_start


    # --------------------------------------------------------
    # Store measurements
    # --------------------------------------------------------

    frame_count += 1

    total_pipeline_time += pipeline_time

    total_yolo_time += yolo_time


    # Instantaneous FPS

    if pipeline_time > 0:

        instant_fps = 1.0 / pipeline_time

    else:

        instant_fps = 0.0


    fps_samples.append(
        instant_fps
    )


    # --------------------------------------------------------
    # Print progress every 5 seconds
    # --------------------------------------------------------

    if current_time - last_report_time >= 5.0:

        elapsed = current_time - start_time

        average_fps = (
            frame_count / elapsed
        )

        print(
            f"Time: {elapsed:5.1f} s | "
            f"Frames: {frame_count:4d} | "
            f"Average FPS: {average_fps:5.2f} | "
            f"Current FPS: {instant_fps:5.2f} | "
            f"YOLO: {yolo_time * 1000:6.1f} ms | "
            f"Objects: {detection_count}"
        )

        last_report_time = current_time


# ============================================================
# FINAL RESULTS
# ============================================================

end_time = time.perf_counter()

actual_duration = (
    end_time - start_time
)

average_fps = (
    frame_count / actual_duration
)

average_pipeline_time = (
    total_pipeline_time / frame_count
)

average_yolo_time = (
    total_yolo_time / frame_count
)

minimum_fps = min(
    fps_samples
)

maximum_fps = max(
    fps_samples
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print()

print("==============================================")
print(" YOLO-WORLD FPS TEST RESULTS")
print("==============================================")
print()

print(
    f"Model                   : {MODEL_PATH}"
)

print(
    f"Resolution              : "
    f"{IMAGE_WIDTH} x {IMAGE_HEIGHT}"
)

print(
    f"Confidence threshold    : "
    f"{CONF_THRES}"
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
print(" YOLO-WORLD FPS TEST COMPLETE")
print("==============================================")
print()


# ============================================================
# STOP CAMERA
# ============================================================

picam2.stop()