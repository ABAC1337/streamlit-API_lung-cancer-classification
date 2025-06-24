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
    page_title="Lung Cancer AI Classifier", 
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .benign-card {
        background: linear-gradient(135deg, #4CAF50, #45a049);
    }
    
    .malignant-card {
        background: linear-gradient(135deg, #f44336, #d32f2f);
    }
    
    .normal-card {
        background: linear-gradient(135deg, #2196F3, #1976D2);
    }
    
    .info-box {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model_once():
    return load_model('CNN_model.h5')

try:
    model = load_model_once()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model file 'CNN_model.h5' not found. Please ensure the model file is in the correct directory.")

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
    """Create an interactive bar chart for probabilities"""
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
    )
    
    return fig

def create_confidence_gauge(confidence):
    """Create a confidence gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Level (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# Main header
st.markdown("""
<div class="main-header">
    <h1>🫁 AI-Powered Lung Cancer Detection</h1>
    <p style="font-size: 1.2em; margin-top: 1rem;">Advanced CT Scan Analysis using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    This AI model analyzes CT scan images to classify lung conditions into three categories:
    
    - **✅ Benign**: Non-cancerous tissue
    - **⚠️ Malignant**: Cancerous tissue  
    - **💙 Normal**: Healthy lung tissue
    
    ---
    
    ### 📋 Instructions:
    1. Upload a CT scan image (JPG, PNG, JPEG)
    2. Wait for AI analysis
    3. Review the prediction and confidence scores
    
    ---
    
    ### ⚠️ Disclaimer:
    This tool is for educational purposes only and should not replace professional medical diagnosis.
    """)
    
    if model_loaded:
        st.success("🤖 AI Model: Ready")
    else:
        st.error("🤖 AI Model: Not Available")

# Main content area
if model_loaded:
    # Upload section
    st.markdown("""
    <div class="upload-section">
        <h2 style="color: #667eea; margin-bottom: 1rem;">📤 Upload CT Scan Image</h2>
        <p>Please upload a clear CT scan image for analysis. Supported formats: JPG, PNG, JPEG</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CT scan image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a high-quality CT scan image for best results"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.subheader("📷 Uploaded Image")
            st.image(image, caption="CT Scan Image", use_container_width=True)
            
            # Image details
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.write(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**File Size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            st.write(f"**Format:** {image.format}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Prediction section
            st.subheader("🧠 AI Analysis Results")
            
            with st.spinner("🔄 Analyzing image..."):
                input_image = preprocess_image(image)
                predictions = model.predict(input_image)[0]
                predicted_index = int(np.argmax(predictions))
                predicted_label = class_names[predicted_index]
                confidence = float(np.max(predictions))
            
            # Main prediction display
            prediction_class = predicted_label.lower()
            st.markdown(f"""
            <div class="prediction-card {prediction_class}-card">
                <h2>{class_emojis[predicted_index]} Prediction: {predicted_label}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            st.plotly_chart(create_confidence_gauge(confidence), use_container_width=True)
        
        # Full-width probability chart
        st.subheader("📊 Detailed Analysis")
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # Probability chart
            fig = create_probability_chart(predictions, class_names)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Probability metrics
            st.subheader("🔍 Class Probabilities")
            for i, (prob, emoji) in enumerate(zip(predictions, class_emojis)):
                st.markdown(f"""
                <div class="metric-container">
                    <h4 style="color: {class_colors[i]};">{emoji} {class_names[i]}</h4>
                    <h3>{prob:.1%}</h3>
                    <div style="background: {class_colors[i]}; height: 8px; border-radius: 4px; width: {prob*100}%;"></div>
                </div>
                """, unsafe_allow_html=True)
        
        # Additional insights
        st.subheader("💡 Insights & Recommendations")
        
        if predicted_label == "Malignant" and confidence > 0.7:
            st.error("⚠️ **High Risk Detected**: The AI model indicates potential malignant tissue. Please consult with a medical professional immediately for proper diagnosis and treatment options.")
        elif predicted_label == "Benign" and confidence > 0.7:
            st.success("✅ **Low Risk**: The analysis suggests benign tissue. However, regular monitoring and medical consultation are still recommended.")
        elif predicted_label == "Normal" and confidence > 0.7:
            st.info("💙 **Healthy Tissue**: The scan appears to show normal lung tissue. Continue with regular health check-ups.")
        else:
            st.warning("🤔 **Uncertain Results**: The model shows lower confidence in this prediction. Consider retaking the scan with better quality or consult a medical professional for further analysis.")

else:
    st.error("🚫 Cannot proceed without the trained model. Please ensure 'CNN_model.h5' is available in the application directory.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🔬 Powered by Deep Learning | 🏥 Healthcare AI Assistant</p>
    <p><small>This application is for educational and research purposes only. Always consult healthcare professionals for medical decisions.</small></p>
</div>
""", unsafe_allow_html=True)