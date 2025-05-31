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

# Custom CSS for better appearance
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

# Load the trained YOLO model
model_path = "best.pt"
model = YOLO(model_path)

# Enhanced title with logo placeholder
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <div style="font-size: 4rem; margin-bottom: 0.5rem;">🚦</div>
    <h1 class="main-title">Traffic Automation System</h1>
    <h3 style="color: #666; font-weight: 300;">Advanced Ambulance Detection & Emergency Response</h3>
</div>
""", unsafe_allow_html=True)

# Enhanced description
st.markdown("""
<div class="description-box">
    <h3>🎯 System Overview</h3>
    <p>This advanced Traffic Automation System utilizes a state-of-the-art YOLOv8 deep learning model to automatically detect ambulances in both static images and real-time video streams. Our system is designed to enhance emergency response capabilities and optimize traffic flow management.</p>
</div>
""", unsafe_allow_html=True)

# Enhanced sidebar
st.sidebar.markdown("## 🔧 System Controls")
st.sidebar.markdown("---")

# Option for image or video upload with enhanced descriptions
app_mode = st.sidebar.selectbox(
    "🎮 Choose Detection Mode", 
    ["Image Detection", "Real-time Video Detection"],
    help="Select your preferred analysis method"
)

st.sidebar.markdown("### 📋 Current Configuration")
st.sidebar.info(f"""
**Detection Mode:** {app_mode}
**Model:** YOLOv8 (best.pt)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚦 Traffic Automation Features")
st.sidebar.markdown("""
- **Emergency Vehicle Priority**
- **Real-time Traffic Analysis** 
- **Automated Signal Control**
- **Multi-lane Monitoring**
- **Data Logging & Reports**
""")

confidence_threshold = 0.7  # Set the confidence threshold to 0.7

if app_mode == "Image Detection":
    st.markdown("## 📷 Static Image Analysis")
    
    # Enhanced upload section
    st.markdown("""
    <div class="feature-highlight">
        <h4>📁 Image Upload Instructions</h4>
        <p>Upload a clear image containing vehicles for ambulance detection. Supported formats: JPG, PNG, JPEG</p>
        <p><strong>💡 Tips for best results:</strong></p>
        <ul>
            <li>Use high-resolution images for better detection accuracy</li>
            <li>Ensure good lighting conditions</li>
            <li>Include full or partial view of vehicles</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload image
    uploaded_file = st.file_uploader(
        "📁 Select Image File", 
        type=["jpg", "png", "jpeg"],
        help="Choose an image file for ambulance detection analysis"
    )

    if uploaded_file is not None:
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🖼️ Original Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image_np = np.array(image)  # Convert to NumPy array

        # Processing indicator
        with st.spinner('🔍 Analyzing image for ambulance detection...'):
            # Run YOLO model
            results = model(image_np)

        # Draw boxes and labels on image
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if conf >= confidence_threshold:  # Only process detections with confidence >= 0.7
                    x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box
                    label = model.names[int(cls)]  # Get class name
                    text = f"{label} {conf:.2f}"  # Label and confidence score
                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_np, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert back to Image format
        result_image = Image.fromarray(image_np)

        with col2:
            st.markdown("### 🎯 Detection Results")
            st.image(result_image, caption="Analysis Complete", use_container_width=True)

        # Enhanced predictions display
        st.markdown("---")
        st.markdown("## 📊 Detection Analysis Report")
        
        predictions_text = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if conf >= confidence_threshold:  # Only display predictions with confidence >= 0.7
                    label = model.names[int(cls)]
                    confidence = conf.item()
                    predictions_text.append(f"🚨 Detected: {label} (Confidence: {confidence:.2%})")
        
        if predictions_text:
            st.success(f"✅ **{len(predictions_text)} Ambulance(s) Successfully Detected!**")
            for pred in predictions_text:
                st.markdown(f'<div class="prediction-item">{pred}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-detection"><strong>ℹ️ Analysis Complete:</strong> No ambulances detected in the current image</div>', unsafe_allow_html=True)

elif app_mode == "Real-time Video Detection":
    st.markdown("## 🎥 Live Video Monitoring")
    
    st.markdown("""
    <div class="feature-highlight">
        <h4>📹 Real-time Detection System</h4>
        <p>This mode activates your webcam for continuous ambulance monitoring. The system will:</p>
        <ul>
            <li><strong>🔴 Live Analysis:</strong> Process video frames in real-time</li>
            <li><strong>⚡ Instant Alerts:</strong> Immediate detection notifications</li>
            <li><strong>📊 Continuous Monitoring:</strong> Non-stop surveillance capability</li>
        </ul>
        <p><strong>⚠️ Note:</strong> Please ensure your webcam is connected and permissions are granted.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Status indicator
    if cap.isOpened():
        st.success("📹 **Camera Status:** Connected and Ready")
    else:
        st.error("❌ **Camera Error:** Unable to access webcam")

    # Streamlit video component
    st.markdown("### 🔴 Live Feed")
    stframe = st.empty()
    
    # Prediction display placeholder
    prediction_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO model on the frame
        results = model(frame)

        # Draw boxes and labels on the frame
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if conf >= confidence_threshold:  # Only process detections with confidence >= 0.7
                    x1, y1, x2, y2 = map(int, box[:4])  # Get bounding box
                    label = model.names[int(cls)]  # Get class name
                    text = f"{label} {conf:.2f}"  # Label and confidence score
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert BGR frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Enhanced predictions display
        predictions_text = []
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if conf >= confidence_threshold:  # Only display predictions with confidence >= 0.7
                    label = model.names[int(cls)]
                    confidence = conf.item()
                    predictions_text.append(f"🚨 ALERT: {label} detected (Confidence: {confidence:.2%})")
        
        # Update predictions display with enhanced styling
        if predictions_text:
            alert_content = "### 🚨 EMERGENCY VEHICLE DETECTED\n" + "\n".join([f"- {pred}" for pred in predictions_text])
            prediction_placeholder.error(alert_content)
        else:
            prediction_placeholder.info("👀 **Monitoring Active:** Scanning for ambulances...")

        # Show video frame in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h3 style="color: #2E86AB; margin-bottom: 1rem;">🚦 Traffic Automation System</h3>
    <p style="color: #666; margin-bottom: 0.5rem;">Powered by YOLOv8 Deep Learning Technology</p>
    <p style="color: #999; font-size: 0.9rem;">Enhancing Emergency Response & Traffic Management</p>
    <div style="margin-top: 1rem;">
        <span style="background-color: #e3f2fd; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem; color: #1976d2;">Real-time Detection</span>
        <span style="background-color: #e8f5e8; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem; color: #388e3c;">Traffic Optimization</span>
        <span style="background-color: #fff3e0; padding: 0.3rem 0.8rem; border-radius: 15px; margin: 0 0.3rem; color: #f57c00;">Emergency Response</span>
    </div>
</div>
""", unsafe_allow_html=True)