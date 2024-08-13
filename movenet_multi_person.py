import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# URL to the MoveNet multi-person model on TensorFlow Hub
MODEL_URL = "https://tfhub.dev/google/movenet/multipose/lightning/1"  # This URL is directly from TensorFlow Hub

# Function to load the MoveNet model
def load_model():
    try:
        model = hub.load(MODEL_URL)
        return model
    except Exception as e:
        print(f"Error loading model from {MODEL_URL}: {e}")
        return None

# Function to draw connections between keypoints and keypoints as dots
def draw_connections_and_keypoints(frame, keypoints, edges):
    h, w, _ = frame.shape
    for person in keypoints[0]:
        keypoints = person[:51].reshape((17, 3))  # Reshape to (17, 3) to get (y, x, confidence) for each keypoint
        for i, keypoint in enumerate(keypoints):
            y, x, c = keypoint
            if c > 0.5:  # Draw only keypoints with confidence > 0.5
                cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 0, 255), -1)  # Draw keypoint as a small circle
        for edge in edges:
            p1, p2 = edge
            y1, x1, c1 = keypoints[p1]
            y2, x2, c2 = keypoints[p2]
            if c1 > 0.5 and c2 > 0.5:
                cv2.line(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)

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

        # Convert the frame to the correct dtype
        inputs = tf.convert_to_tensor([resized_frame], dtype=tf.int32)
        
        # Perform inference
        outputs = model.signatures["serving_default"](inputs)

        # Extract keypoints and draw connections and keypoints
        keypoints_with_scores = outputs['output_0'].numpy()
        print("Keypoints with scores shape:", keypoints_with_scores.shape)
        print("Keypoints with scores:", keypoints_with_scores)

        draw_connections_and_keypoints(frame, keypoints_with_scores, KEYPOINT_EDGES)

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

if __name__ == "__main__":
    # Load the MoveNet model
    model = load_model()

    if model:
        # Run pose detection on webcam feed
        run_pose_detection(model)
    else:
        print("Failed to load the MoveNet model. Please check the URL and try again.")
