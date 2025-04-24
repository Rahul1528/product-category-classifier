import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input # Import specific preprocess_input
import numpy as np
from PIL import Image
import os
st.set_page_config(page_title="E-Commerce Product Classifier", layout="wide")
# --- Configuration ---
MODEL_PATH = 'product_classifier_model.h5'
IMG_SIZE = (224, 224) # Must match the input size used during training
# IMPORTANT: Make sure this order matches the training output's class indices
# Check the output of `train_generator.class_indices` in your training script
CLASS_NAMES = ['Clothing', 'Footwear', 'Electronics']

# --- Model Loading ---
# Cache the model loading function to avoid reloading on every interaction
@st.cache_resource
def load_model(model_path):
    """Loads the Keras model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # You might want to stop the app execution here if the model fails to load
        # For example: st.stop()
        return None

model = load_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(img_pil):
    """Prepares a PIL image for the model."""
    img = img_pil.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    # Check if image is grayscale and convert to RGB if necessary
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array] * 3, axis=-1)
    # Ensure image has 3 channels if it's RGBA
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    img_array_expanded = np.expand_dims(img_array, axis=0)
    # Use the specific preprocessing function for MobileNetV2
    return preprocess_input(img_array_expanded)

# --- Streamlit App UI ---

st.title("📦 E-Commerce Smart Product Image Classifier")
st.markdown("""
    Upload one or more product images, and the app will classify them into
    categories like Clothing, Footwear, or Electronics.
""")

# File Uploader
uploaded_files = st.file_uploader(
    "📤 Upload product images:",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

st.markdown("---") # Separator

if model is None:
    st.warning("Model could not be loaded. Please check the model file and path.")
elif uploaded_files:
    st.subheader("🖼️ Classification Results:")

    # Display images and predictions in columns for better layout
    num_columns = 3 # Adjust as needed
    cols = st.columns(num_columns)
    col_index = 0

    total_images = len(uploaded_files)
    classified_count = 0

    for uploaded_file in uploaded_files:
        try:
            # Open the image
            img_pil = Image.open(uploaded_file)

            # Preprocess the image
            processed_img = preprocess_image(img_pil)

            # Make prediction
            predictions = model.predict(processed_img)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(predictions[0]) * 100 # Get confidence score

            # Display in the next available column
            with cols[col_index]:
                st.image(img_pil, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                st.success(f"🔎 Predicted: **{predicted_class_name}** ({confidence:.2f}%)")
                st.markdown("---") # Separator between images in a column

            col_index = (col_index + 1) % num_columns # Move to the next column index
            classified_count += 1

        except Exception as e:
             with cols[col_index]:
                st.error(f"Error processing {uploaded_file.name}: {e}")
                st.markdown("---")
             col_index = (col_index + 1) % num_columns

    st.sidebar.success(f"✅ Total Images Uploaded: {total_images}")
    st.sidebar.info(f"✅ Images Classified: {classified_count}")


else:
    st.info("Awaiting image uploads...")

# --- Optional Footer ---
st.markdown("---")
st.markdown("Built with TensorFlow & Streamlit")