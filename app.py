import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Pothole Detector")
st.title("🕳Pothole Detector (YOLOv8)")
st.write(
    f"Upload a street image, and the YOLOv8s model will try to detect potholes. [Project description in notebook]")

MODEL_NAME = "best.pt"

model = YOLO(MODEL_NAME)

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, caption='Uploaded image.', use_column_width=True)
        st.write("")

        if st.button('Run detection'):
            results = model(image)

            result_image = results[0].plot()

            result_image_rgb = result_image[..., ::-1]

            st.subheader("Detection Results:")
            st.image(result_image_rgb, caption='Image with detected objects.', use_column_width=True)

            st.write("Detected objects (summary):")
            st.dataframe(results[0].summary())
else:
    st.warning("Model could not be loaded correctly. The application cannot perform detection.")

st.sidebar.header("About the project")
st.sidebar.write(f"""
This is an application demonstrating a YOLOv8 model] trained to detect potholes.

The project involved manual data labeling (in Label Studio)] and training the model in the cloud (Google Colab)].

**Author:** Oskar Andrukiewicz
\n[GitHub Repository](https://github.com/gulis-dev/Pothole-Detector)
\n[LinkedIn](https://www.linkedin.com/in/oskar-andrukiewicz/)
""")