import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np

st.title("🌊 Marine Debris Detection Dashboard")

model = YOLO("yolov8n.pt")  # Using pretrained small model

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)
    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Detection Result", use_column_width=True)

    st.write("### Detection Summary")
    boxes = results[0].boxes
    if boxes:
        classes = boxes.cls.cpu().numpy()
        unique, counts = np.unique(classes, return_counts=True)
        for u, c in zip(unique, counts):
            st.write(f"Class {int(u)}: {int(c)} objects")
