import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Page configuration
st.set_page_config(
    page_title="Traffic Automation System",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .description-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-highlight {
        background-color: #f0f8ff;
        padding: 1rem;
        border-left: 4px solid #2E86AB;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-item {
        background-color: #e8f5e8;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
    .no-detection {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #ffc107;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Load YOLOv8 model
model_path = "best.pt"
model = YOLO(model_path)
confidence_threshold = 0.7

# Top Title
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div style="font-size: 4rem; margin-bottom: 0.5rem;">🚦</div>
    <h1 class="main-title">Traffic Automation System</h1>
    <h3 style="color: #666; font-weight: 300;">Advanced Ambulance Detection & Emergency Response</h3>
</div>
""", unsafe_allow_html=True)

# System Overview
st.markdown("""
<div class="description-box">
    <h3>🎯 System Overview</h3>
    <p>This advanced Traffic Automation System utilizes a state-of-the-art YOLOv8 deep learning model to automatically detect ambulances in both static images and real-time video streams. Our system is designed to enhance emergency response capabilities and optimize traffic flow management.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🔧 System Controls")
app_mode = st.sidebar.selectbox("🎮 Choose Detection Mode", ["Image Detection", "Real-time Video Detection"])
st.sidebar.markdown("### 📋 Current Configuration")
st.sidebar.info(f"**Detection Mode:** {app_mode}\n\n**Model:** YOLOv8 (best.pt)")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 Traffic Automation Features")
st.sidebar.markdown("""
- **Emergency Vehicle Priority**
- **Real-time Traffic Analysis**
- **Automated Signal Control**
- **Multi-lane Monitoring**
- **Data Logging & Reports**
""")

# ---------------- IMAGE DETECTION -------------------
if app_mode == "Image Detection":
    st.markdown("## 📷 Static Image Analysis")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            image_np = np.array(image)

            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                results = model(image_bgr)
                display_image = image_np.copy()

                predictions_text = []

                for result in results:
                    if hasattr(result.boxes, 'xyxy') and len(result.boxes.xyxy) > 0:
                        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                            if conf >= confidence_threshold:
                                x1, y1, x2, y2 = map(int, box[:4])
                                label = model.names[int(cls)]
                                text = f"{label} {conf:.2f}"
                                predictions_text.append(f"<div class='prediction-item'>🚑 Detected: {label} (Confidence: {conf:.2f})</div>")
                                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                result_image = Image.fromarray(display_image)
                st.image(result_image, caption="🔍 Detection Result", use_container_width=True)

                if predictions_text:
                    st.markdown("<h4>📌 Predictions:</h4>" + "".join(predictions_text), unsafe_allow_html=True)
                else:
                    st.markdown("<div class='no-detection'>🚫 No ambulances detected</div>", unsafe_allow_html=True)
            else:
                st.error("Invalid image format. Please upload a valid RGB image.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ---------------- VIDEO DETECTION -------------------
elif app_mode == "Real-time Video Detection":
    st.markdown("## 🎥 Real-time Video Stream")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("🚫 Could not open webcam. Please check your camera connection.")
    else:
        stframe = st.empty()
        pred_placeholder = st.empty()
        stop_button = st.button("🛑 Stop Video")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Could not read frame from webcam.")
                break

            try:
                results = model(frame)
                predictions_text = []

                for result in results:
                    if hasattr(result.boxes, 'xyxy') and len(result.boxes.xyxy) > 0:
                        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                            if conf >= confidence_threshold:
                                x1, y1, x2, y2 = map(int, box[:4])
                                label = model.names[int(cls)]
                                text = f"{label} {conf:.2f}"
                                predictions_text.append(f"🚑 {label} (Confidence: {conf:.2f})")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                if predictions_text:
                    pred_placeholder.markdown("### 🚑 Detected Ambulance(s):\n" + "\n".join(predictions_text))
                else:
                    pred_placeholder.markdown("<div class='no-detection'>🚫 No ambulances detected</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during video processing: {str(e)}")
                break

        cap.release()
