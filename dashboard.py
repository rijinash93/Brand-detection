"""
🎥 Brand & Logo Detection (YOLOv8)
Run this with:
    streamlit run brand_logo_detection.py
"""

import streamlit as st
import cv2
import os
import tempfile
import pandas as pd
from ultralytics import YOLO

# -----------------------------------------
# Page Setup
# -----------------------------------------
st.set_page_config(page_title="Brand & Logo Detection", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🧠 Brand & Logo Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect TATA, BMW, Ford, and more using your YOLOv8 custom model</p>', unsafe_allow_html=True)

# -----------------------------------------
# Load YOLO Model (custom or default)
# -----------------------------------------
@st.cache_resource
def load_yolo_model():
    try:
        if os.path.exists("best.pt"):
            model_path = "best.pt"
        else:
            model_path = "yolov8n.pt"
        st.info(f"📦 Loading model: {model_path}")
        model = YOLO(model_path)
        st.success(f"✅ Model loaded: {model_path}")
        return model, model_path
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None, None

model, model_path = load_yolo_model()
if not model:
    st.stop()

# -----------------------------------------
# Upload Video
# -----------------------------------------
uploaded_file = st.file_uploader("📤 Upload a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    # Read video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"🎞️ {uploaded_file.name} | {total_frames} frames | {fps:.1f} FPS | {width}x{height}px")

    # -----------------------------------------
    # Process Frames
    # -----------------------------------------
    stframe = st.empty()
    progress = st.progress(0, text="Processing video...")
    detections = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:  # every 5th frame for speed
            results = model(frame, conf=0.25)
            annotated = results[0].plot()
            stframe.image(annotated, channels="BGR", use_container_width=True)

            # store results
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls]
                    t = round(frame_count / fps, 2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    detections.append({
                        "Frame": frame_count,
                        "Timestamp (s)": t,
                        "Label": label,
                        "Confidence": round(conf, 2),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })

        frame_count += 1
        progress.progress(min(frame_count / total_frames, 1.0), text=f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    os.unlink(video_path)
    progress.empty()

    # -----------------------------------------
    # Show Results
    # -----------------------------------------
    st.divider()
    if detections:
        df = pd.DataFrame(detections)
        st.success(f"✅ Detection completed with {len(df)} detections using {model_path}.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Frames Processed", frame_count)
        col2.metric("Total Detections", len(df))
        col3.metric("Unique Labels", df['Label'].nunique())

        st.subheader("📊 Detection Summary")
        summary = df.groupby("Label")["Confidence"].agg(["count", "mean"]).reset_index()
        summary.columns = ["Label", "Count", "Avg Confidence"]
        st.dataframe(summary, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download Detections CSV",
            csv,
            f"{uploaded_file.name}_detections.csv",
            "text/csv"
        )

    else:
        st.warning("⚠️ No brands or objects detected.")
else:
    st.info("👆 Upload a video file to start detection.")
