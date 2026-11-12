import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Webcam", frame)

    # Press 'q' or 'ESC' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()