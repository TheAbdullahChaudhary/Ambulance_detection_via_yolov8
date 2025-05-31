import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load the trained YOLO model
model_path = "best.pt"
model = YOLO(model_path)

st.title("🚑 YOLO Ambulance Detection")

# Add a description to the Streamlit app
st.markdown("""
This app uses a trained YOLOv8 model to detect ambulances in images and video.
You can upload an image or use your webcam to perform real-time detection.
""")

# Option for image or video upload
app_mode = st.sidebar.selectbox("Choose the app mode", ["Image Detection", "Real-time Video Detection"])

if app_mode == "Image Detection":
    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)  # Convert to NumPy array

        # Run YOLO model
        results = model(image_np)

        # Draw boxes and labels on image
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box
                label = model.names[int(cls)]  # Get class name
                text = f"{label} {conf:.2f}"  # Label and confidence score
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to Image format
        result_image = Image.fromarray(image_np)

        # Add predictions display
        st.subheader("Predictions:")
        predictions_text = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label = model.names[int(cls)]
                confidence = conf.item()
                predictions_text.append(f"Detected: {label} (Confidence: {confidence:.2f})")
        if predictions_text:
            for pred in predictions_text:
                st.write(pred)
        else:
            st.write("No ambulances detected")

        # Show result
        st.image(result_image, caption="Detected Image", use_container_width=True)

elif app_mode == "Real-time Video Detection":
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Streamlit video component
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)

        # Draw boxes and labels on the frame
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box
                label = model.names[int(cls)]  # Get class name
                text = f"{label} {conf:.2f}"  # Label and confidence score
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add predictions display
        predictions_text = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                label = model.names[int(cls)]
                confidence = conf.item()
                predictions_text.append(f"Detected: {label} (Confidence: {confidence:.2f})")
        # Create a placeholder for predictions
        pred_placeholder = st.empty()
        if predictions_text:
            pred_placeholder.write("\n".join(predictions_text))
        else:
            pred_placeholder.write("No ambulances detected")

        # Show video frame in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()