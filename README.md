# 🚑 YOLOv8 Ambulance Detection App

![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-orange?style=for-the-badge&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-blueviolet?style=for-the-badge&logo=pytorch)
![OpenCV](https://img.shields.io/badge/image%20processing-OpenCV-green?style=for-the-badge&logo=opencv)

> Detect ambulances in images or in real-time using your webcam, powered by a custom-trained YOLOv8 model and an intuitive Streamlit interface.

---

## 📸 Features

- ✅ Upload and detect ambulances in static images
- 🎥 Real-time webcam-based ambulance detection
- ⚡ Powered by YOLOv8 for ultra-fast inference
- 🧠 Smart bounding boxes and class confidence display
- 💡 Built using Python, Streamlit, OpenCV, and PyTorch

---

## 🛠 How to Run

### 🔧 Prerequisites

Make sure you have the following installed:
- Python 3.8+
- pip
- [YOLOv8 dependencies](https://docs.ultralytics.com/)
- Streamlit
- OpenCV
- Pillow
- PyTorch
- `ultralytics` library

### 🧪 Installation

```bash
# Clone the repo
git clone https://github.com/your-username/ambulance-detection-app.git
cd ambulance-detection-app

# Install dependencies
pip install -r requirements.txt
```

### ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📂 File Structure

```
📁 ambulance-detection-app/
🔹 app.py              # Streamlit frontend & detection logic
🔹 best.pt             # Trained YOLOv8 model weights
🔹 requirements.txt    # Python dependencies
🔹 README.md           # This file
```

---

## 🧠 Model

This app uses a **YOLOv8** model trained specifically to detect **ambulances** from various angles. The model (`best.pt`) is loaded with the `ultralytics` library.

Training was done on a custom dataset of over **2,200 annotated images** containing ambulances.

---

## 🤖 Modes

- **Image Detection:** Upload any `.jpg`, `.jpeg`, or `.png` image and detect ambulances instantly.
- **Real-time Detection:** Activate your webcam and see live detections with bounding boxes and class confidence scores.

---

## 📟 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Streamlit](https://streamlit.io) for the beautiful web framework
- [OpenCV](https://opencv.org/) for image manipulation
"# Ambulance_detection_via_yolov8" 
