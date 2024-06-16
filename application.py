import streamlit as st
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from PIL import Image

def main():
    st.title("Animal Classifier")
    st.write("Upload an image and see it displayed. and then proceed to classification.")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "WebP"])

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        image_resized = image.resize((224, 224), Image.BICUBIC)  # Using BICUBIC resampling filter
        st.image(image_resized, caption='Uploaded Image.', width=400)  # Slightly increased width

        # Option for classification
        classify = st.button("Proceed to Classification")
        if classify:
            # Load the trained model
            model_path = 'Custom_CNN_Animal_classifier_model.h5'
            model = load_model(model_path)

            # Function to preprocess the image
            def preprocess_image(image):
                image = img_to_array(image) / 255.0
                image = np.expand_dims(image, axis=0)
                return image

            # Predict the uploaded image
            preprocessed_image = preprocess_image(image_resized)
            prediction = model.predict(preprocessed_image)
            categories = ['Domestic', 'Wild']
            category_index = np.argmax(prediction)
            probability = prediction[0][category_index]
            category = categories[category_index]
            st.write(f"Prediction: {category} (Probability: {probability:.2f})")

if __name__ == '__main__':
    main()
