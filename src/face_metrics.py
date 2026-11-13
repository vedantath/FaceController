"""
a diagnostic view that displays:
1. Left eye blink score.
2. Right eye blink score.
3. Head pitch, yaw, and roll.
4. Iris positions.
"""
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

from gesture_detection import GestureDetector

# Using MediaPipe's facial transformation matrix for head orientation and its blendshapes for blink detection

# Note: MediaPipe returns 52 blendshape scores, including eyeBlinkLeft and eyeBlinkRight. These are much easier to work with than manually calculating blink ratios.

# Goal: a live dashboard where you can wink with either eye and see the corresponding score change independently

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions  
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

detector = GestureDetector()

model_path = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

# Create face landmarker instance
landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit()
    raise RuntimeError("Could not access webcam")

def get_blendshapes(result):
    """Extract facial expression scores"""
    if not result.face_blendshapes:
        return {}
    return {item.category_name: item.score for item in result.face_blendshapes[0]}
    
def get_head_pose(result):
    """Estimate pitch, yaw, and roll from transformation matrix."""
    if not result.facial_transformation_matrixes:
        return None
    
    M = np.array(result.facial_transformation_matrixes[0]).reshape(4, 4)
    rot = M[:3, :3]
    angles = cv2.RQDecomp3x3(rot)[0]  # Returns pitch, yaw, roll in degrees
    pitch, yaw, roll = angles
    return pitch, yaw, roll
    
    
def draw_text(frame, text, position, color=(0, 255, 0)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = time.monotonic_ns() // 1_000_000
        result = landmarker.detect_for_video(
            mp_image, timestamp
        )

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            h, w, _ = frame.shape

            # Draw all facial landmarks
            for point in landmarks:
                x = int(point.x * w)
                y = int(point.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Draw iris centers
            for index in [468, 473]:
                point = landmarks[index]
                x = int(point.x * w)
                y = int(point.y * h)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Extract blink scores
            blendshapes = get_blendshapes(result)

            # The mirrored preview swaps the viewer's left and right sides.
            left_blink = blendshapes.get("eyeBlinkRight", 0)
            right_blink = blendshapes.get("eyeBlinkLeft", 0)

            draw_text(
                frame,
                f"Left blink: {left_blink:.2f}",
                (20, 35)
            )

            draw_text(
                frame,
                f"Right blink: {right_blink:.2f}",
                (20, 65)
            )

            # Preliminary wink detection
            if left_blink > 0.6 and right_blink < 0.3:
                draw_text(frame, "LEFT WINK", (20, 105), (0, 255, 255))

            elif right_blink > 0.6 and left_blink < 0.3:
                draw_text(frame, "RIGHT WINK", (20, 105), (0, 255, 255))

            # Head pose
            pose = get_head_pose(result)

            if pose is not None:
                pitch, yaw, roll = pose

                events = detector.update(
                    left_blink,
                    right_blink,
                    roll
                )

                for event in events:
                    print(f"GESTURE DETECTED: {event}")

                draw_text(
                    frame,
                    f"Pitch: {pitch:.1f}",
                    (20, 155)
                )

                draw_text(
                    frame,
                    f"Yaw: {yaw:.1f}",
                    (20, 185)
                )

                draw_text(
                    frame,
                    f"Roll: {roll:.1f}",
                    (20, 215)
                )

        cv2.imshow("Facial Metrics", frame)

        # Exit on 'q' or "ESC" key pressed
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:
            break
    
finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    