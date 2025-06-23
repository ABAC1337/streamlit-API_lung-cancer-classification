import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
@st.cache_resource
def load_model_once():
    return load_model('CNN_model.h5')

model = load_model_once()
class_names = ['Benign', 'Malignant', 'Normal']

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    img_array = np.array(image)
    resized_image = cv2.resize(img_array, (150, 150))
    input_image = resized_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

st.set_page_config(page_title="Lung Cancer Classifier", page_icon="🫁")
st.title("🫁 Lung Cancer Prediction from CT Scan Images")

# Option 1: Upload one image
st.header("📤 Upload Single Image")
uploaded_file = st.file_uploader("Upload a CT image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_image = preprocess_image(image)
    predictions = model.predict(input_image)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]

    st.markdown(f"### 🧠 Prediction: **{predicted_label}**")
    st.markdown("#### 🔍 Class Probabilities:")
    for i, prob in enumerate(predictions):
        st.write(f"{class_names[i]}: {prob:.4f}")

