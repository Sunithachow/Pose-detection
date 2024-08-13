import cv2
import mediapipe as mp

# Initialize MediaPipe pose solution
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to draw landmarks
def draw_landmarks(image, landmarks, connections):
    if landmarks:
        for idx, landmark in enumerate(landmarks.landmark):
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        if connections:
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]
                start_point = (int(start.x * image.shape[1]), int(start.y * image.shape[0]))
                end_point = (int(end.x * image.shape[1]), int(end.y * image.shape[0]))
                cv2.line(image, start_point, end_point, (0, 255, 0), 2)

# Load video or webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect poses
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract bounding box of detected poses
        x_coords = []
        y_coords = []
        for landmark in results.pose_landmarks.landmark:
            x_coords.append(landmark.x * frame.shape[1])
            y_coords.append(landmark.y * frame.shape[0])

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

