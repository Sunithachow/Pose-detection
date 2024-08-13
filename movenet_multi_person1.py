import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import math

# URL to the MoveNet multi-person model on TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"

# Function to load the MoveNet model
def load_model():
    try:
        model = hub.load(MODEL_URL)
        return model
    except Exception as e:
        print(f"Error loading model from {MODEL_URL}: {e}")
        return None

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to draw connections between keypoints, keypoints as dots, and calculate angles
def draw_connections_keypoints_and_angles(frame, keypoints_with_scores, edges):
    h, w, _ = frame.shape
    keypoints = keypoints_with_scores[:, :, :2]  # Extract keypoint positions
    confidence_scores = keypoints_with_scores[:, :, 2]  # Extract confidence scores
    
    for person in range(keypoints.shape[0]):
        for i, keypoint in enumerate(keypoints[person]):
            y, x = keypoint
            c = confidence_scores[person, i]
            if c > 0.5:  # Draw only keypoints with confidence > 0.5
                cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)  # Draw keypoint as a small circle

        for edge in edges:
            p1, p2 = edge
            if p1 < keypoints.shape[1] and p2 < keypoints.shape[1]:  # Check if indices are within bounds
                y1, x1 = keypoints[person, p1]
                y2, x2 = keypoints[person, p2]
                c1 = confidence_scores[person, p1]
                c2 = confidence_scores[person, p2]
                if c1 > 0.5 and c2 > 0.5:
                    cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)

        # Example angles: Shoulder, Elbow, and Wrist (for left and right arms)
        for triplet in ANGLE_KEYPOINTS:
            if all(idx < keypoints.shape[1] for idx in triplet):  # Check if all indices are within bounds
                a = keypoints[person, triplet[0]].astype(int) * [h, w]
                b = keypoints[person, triplet[1]].astype(int) * [h, w]
                c = keypoints[person, triplet[2]].astype(int) * [h, w]
                ca = confidence_scores[person, triplet[0]]
                cb = confidence_scores[person, triplet[1]]
                cc = confidence_scores[person, triplet[2]]
                if ca > 0.5 and cb > 0.5 and cc > 0.5:
                    angle = calculate_angle(a, b, c)
                    cv2.putText(frame, str(int(angle)), tuple(b), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

# Function to run pose detection on a single frame
def run_pose_detection(model):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to expected model input dimensions
        resized_frame = cv2.resize(frame, (256, 256))
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Convert the resized frame to the correct dtype (int32)
        inputs = tf.convert_to_tensor([resized_frame], dtype=tf.int32)
        
        # Perform inference
        outputs = model.signatures['serving_default'](input=inputs)

        # Extract keypoints and draw connections, keypoints, and angles
        keypoints_with_scores = outputs['output_0'].numpy()

        # Display keypoints on the frame
        draw_connections_keypoints_and_angles(frame, keypoints_with_scores, KEYPOINT_EDGES)

        # Display the frame
        cv2.imshow('Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Define the edges to be connected for drawing
KEYPOINT_EDGES = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [7, 9],
                  [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13],
                  [13, 15], [12, 14], [14, 16]]

# Define keypoints for angle calculation (shoulder, elbow, wrist)
# Indices: 5 (left shoulder), 7 (left elbow), 9 (left wrist)
# Indices: 6 (right shoulder), 8 (right elbow), 10 (right wrist)
ANGLE_KEYPOINTS = [(5, 7, 9), (6, 8, 10)]

if __name__ == "__main__":
    # Load the MoveNet model
    model = load_model()

    if model:
        # Run pose detection on webcam feed
        run_pose_detection(model)
    else:
        print("Failed to load the MoveNet model. Please check the URL and try again.")
