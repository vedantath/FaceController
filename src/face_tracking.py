import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True
)

landmarker = FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create a MediaPipe Image object from the RGB frame
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 33 # Assuming a frame rate of ~30 FPS, increment timestamp by 33 milliseconds for each frame
    result = landmarker.detect_for_video(mp_image, timestamp) # Detect face landmarks in the video frame


    # Draw the detected face landmarks on the frame
    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        h, w, _ = frame.shape

        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)   # small green circle for each landmark

        cv2.putText(
            frame,
            f"Landmarks: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Tracking", frame)

    # Press 'q' or "ESC" to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()