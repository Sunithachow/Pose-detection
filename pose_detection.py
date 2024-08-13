import cv2
import mediapipe as mp
import numpy as np

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

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

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

        # Calculate angles between keypoints
        shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1],
                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]]
        elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * frame.shape[1],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * frame.shape[0]]
        wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1],
                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]]

        angle = calculate_angle(shoulder, elbow, wrist)
        cv2.putText(frame, str(angle), tuple(np.multiply(elbow, [frame.shape[1], frame.shape[0]]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Pose Detection', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
