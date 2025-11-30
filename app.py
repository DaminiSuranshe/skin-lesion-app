import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from PIL import Image
import cv2
import os

# Page configuration
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🩺",
    layout="wide"
)

# App title
st.title("🩺 Skin Lesion Classification (Benign vs Malignant)")
st.write("Upload a skin lesion image, and the AI model will classify it.")
st.write("---")
# ---------------------------
# LOAD MODEL
# ---------------------------
@st.cache_resource
def load_cnn_model():
    model = load_model("best_skin_lesion_model.h5")
    return model

model = load_cnn_model()
# ---------------------------
# IMAGE PREPROCESSING
# ---------------------------
def preprocess_image(img):
    # Resize to model input size
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array
# ---------------------------
# GRAD-CAM IMPLEMENTATION
# ---------------------------
def grad_cam(img_path, model, layer_name="Conv_1"):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model that outputs (last conv layer, predictions)
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
    
        # handle both (1,) and (1,1)
        loss = predictions[..., 0]


    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs[0].shape[:2], dtype=np.float32)

    # Weighted sum of activations
    for i, w in enumerate(weights):
        cam += w * conv_outputs[0][:, :, i]

    # Normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    cam = cv2.resize(cam.numpy(), (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = heatmap * 0.4 + np.array(img) * 0.6
    return img, heatmap, overlay.astype(np.uint8)
# ---------------------------

# ---------------------------
# STREAMLIT USER INTERFACE
# ---------------------------

uploaded_file = st.file_uploader("📤 Upload a skin lesion image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    st.subheader("📍 Uploaded Image")
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    # Preprocess image
    img_array = preprocess_image(img)

    # Make prediction
    prob = model.predict(img_array)[0][0]
    pred_label = "Malignant" if prob > 0.5 else "Benign"
    confidence = prob if prob > 0.5 else (1 - prob)

    st.write("---")

    # Display prediction
    st.subheader("🔍 Prediction")
    st.write(f"### **Result: {pred_label}**")
    st.write(f"### **Confidence: {confidence*100:.2f}%**")

    st.write("---")
    st.subheader("🔬 Model Explanation (Grad-CAM)")

    # Save uploaded image temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Generate Grad-CAM visualizations
    original, heatmap, overlay = grad_cam("temp.jpg", model)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(original, caption="Original Image", width=250)

    with col2:
        st.image(heatmap, caption="Grad-CAM Heatmap", width=250)

    with col3:
        st.image(overlay, caption="Overlay (Heatmap on Image)", width=250)
