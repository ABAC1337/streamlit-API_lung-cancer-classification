import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Lung Cancer Classification Tools", 
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="auto"
)

# Sidebar navigation
page = st.sidebar.selectbox("Navigation", ["Home", "Tools", "Documentation", "About"])

# Load model
@st.cache_resource
def load_model_once():
    try:
        return load_model('Model_CNN_FIK.h5')
    except:
        return None

model = load_model_once()
model_loaded = model is not None
class_names = ['Benign', 'Malignant', 'Normal']
class_colors = ['#4CAF50', '#f44336', '#2196F3']
class_emojis = ['✅', '⚠️', '💙']

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    img_array = np.array(image)
    resized_image = cv2.resize(img_array, (150, 150))
    input_image = resized_image.astype("float32") / 255.0
    input_image = np.expand_dims(input_image, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def create_probability_chart(predictions, class_names):
    fig = go.Figure(data=[
        go.Bar(
            x=class_names,
            y=predictions,
            marker_color=class_colors,
            text=[f'{p:.1%}' for p in predictions],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Classification Probabilities",
        xaxis_title="Class",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_confidence_gauge(confidence):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)", 'font': {'color': 'white'}},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#00d4ff"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(255,255,255,0.1)"},
                {'range': [50, 80], 'color': "rgba(255,255,0,0.3)"},
                {'range': [80, 100], 'color': "rgba(0,255,0,0.3)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

# Page routing
if page == "Home":
    st.title("Welcome to Lung Cancer Classification Tools")
    st.write("This tool uses deep learning to classify lung CT scans into Benign, Malignant, or Normal.")

elif page == "Tools":
    st.header("Upload CT Scan")
    if model_loaded:
        uploaded_file = st.file_uploader("Choose a CT scan image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="CT Scan Image", use_container_width=True)
            input_image = preprocess_image(image)
            with st.spinner("Analyzing image..."):
                predictions = model.predict(input_image)[0]
                predicted_index = int(np.argmax(predictions))
                predicted_label = class_names[predicted_index]
                confidence = float(np.max(predictions))

            if confidence < 0.8:
                st.error("❌ Prediction confidence is too low. The image might be unsuitable or unclear. Please upload a better quality CT scan.")
            else:
                st.subheader(f"Prediction: {class_emojis[predicted_index]} {predicted_label}")
                st.write(f"Confidence: {confidence:.1%}")
                st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
                st.plotly_chart(create_probability_chart(predictions, class_names), use_container_width=True)

    else:
        st.error("Model file not found. Please ensure 'Model_CNN_FIK.h5' is in the working directory.")

elif page == "Documentation":
    st.header("Model Documentation")
    st.markdown("""
    - **Model Architecture**: CNN with multiple convolutional layers and dropout.
    - **Input Size**: 150x150 grayscale image
    - **Classes**: Benign, Malignant, Normal
    - **Framework**: TensorFlow / Keras
    - **Accuracy**: High accuracy on validation data
    - **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
    """)

elif page == "About":
    st.header("About This Tool")
    st.write("""
    This tool is developed for educational and research purposes. It uses deep learning (CNN)
    to classify lung cancer from CT scan images.

    **Disclaimer**: This tool is not intended for medical use. Always consult a healthcare professional.
    """)
