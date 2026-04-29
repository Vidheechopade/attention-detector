import os
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu"
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import os

# --- Setup MediaPipe & YOLO ---
@st.cache_resource
def load_models():
    # Get the directory where app.py is located to create absolute paths
    base_path = os.path.dirname(__file__)
    
    weights_path = os.path.join(base_path, "yolo", "yolov3.weights")
    cfg_path = os.path.join(base_path, "yolo", "yolov3.cfg")
    names_path = os.path.join(base_path, "yolo", "coco.names")

    # MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
    
    # YOLO Network
    net = cv2.dnn.readNet(weights_path, cfg_path)
    
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    # Handle different versions of OpenCV return types for unconnected layers
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers[0], list) or isinstance(unconnected_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_layers]
    
    return face_mesh, net, classes, output_layers

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    vertical1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
    vertical2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
    horizontal = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
    return (vertical1 + vertical2) / (2.0 * horizontal)

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Attention Detector", layout="wide")
st.title("Student Attention Detector")
st.write("Real-time monitoring using Computer Vision (YOLOv3 & MediaPipe)")

# Sidebar for controls
st.sidebar.header("Settings")
run = st.sidebar.checkbox('Start Camera Feed')
confidence_threshold = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.5)

# Placeholders for the video feed and live metrics
col1, col2 = st.columns([2, 1])
with col1:
    frame_placeholder = st.empty()
with col2:
    st.subheader("Live Metrics")
    status_text = st.empty()
    score_metric = st.empty()

# Constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Initialize Session State for score persistence
if 'attention_score' not in st.session_state:
    st.session_state.attention_score = 100.0

# Load models
try:
    face_mesh, net, classes, output_layers = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

if run:
    cap = cv2.VideoCapture(0) # 0 is typically the built-in webcam
    
    while cap.isOpened() and run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame. Check if webcam is in use by another app.")
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "Focused"
        phone_detected = False

        # --- YOLO Detection ---
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold and classes[class_id] == "cell phone":
                    phone_detected = True

        # --- Attention Logic ---
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Eye EAR
            ear = (eye_aspect_ratio(landmarks, LEFT_EYE, w, h) + eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)) / 2
            
            if ear < 0.20:
                status = "Sleepy"
                st.session_state.attention_score -= 1
            else:
                st.session_state.attention_score += 0.05 # Slower recovery
                
            # Head Position
            nose_x = int(landmarks[1].x * w)
            if nose_x < w * 0.35 or nose_x > w * 0.65:
                status = "Looking Away"
                st.session_state.attention_score -= 0.5
        else:
            status = "No Face Detected"
            st.session_state.attention_score -= 1

        if phone_detected:
            status = "Using Phone"
            st.session_state.attention_score -= 2

        # Clamp score
        st.session_state.attention_score = max(0, min(100, st.session_state.attention_score))

        # --- UI Updates ---
        # Alert color logic
        if st.session_state.attention_score < 50:
            alert_color = (255, 0, 0) # Red
            display_status = "⚠️ NOT FOCUSED"
        else:
            alert_color = (0, 255, 0) # Green
            display_status = "✅ FOCUSED"

        # Draw info on the frame
        cv2.putText(frame, f"Status: {status}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, alert_color, 2)
        if phone_detected:
            cv2.putText(frame, "PHONE DETECTED", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Convert to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update sidebar/metrics
        status_text.markdown(f"### {display_status}")
        score_metric.metric("Attention Score", f"{int(st.session_state.attention_score)}%")

    cap.release()
else:
    st.info("The camera is currently stopped. Check the 'Start Camera Feed' box in the sidebar to begin.")