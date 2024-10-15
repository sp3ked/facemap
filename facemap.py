import cv2
import mediapipe as mp

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Set up camera (webcam)
cap = cv2.VideoCapture(0)  # '0' selects the default camera, change if using external camera

# Initialize the face detection model
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()  # Read frame from the camera
        if not success:
            print("Ignoring empty ca frame.")
            continue

        # Convert the image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect faces
        results = face_detection.process(image_rgb)

        # Draw face detections on the image
        if results.detections:
            for detection in results.detections:
                # Draw the face landmark results on the image
                mp_drawing.draw_detection(image, detection)

        # Display the output
        cv2.imshow('Face Detection', image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
