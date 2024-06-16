import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st

# Load the pre-trained model
saved_model_path = 'Custom_CNN_Animal_classifier_model.h5'
loaded_model = load_model(saved_model_path)

# Define width and height for resizing frames
width, height = 224, 224

# Define the categories for classification
categories = ['Domestic', 'Wild']

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (width, height))
        frame = frame / 255.0
        frames.append(frame)
    cap.release()

    frames = np.array(frames)
    predictions = loaded_model.predict(frames)

    # Aggregate predictions over frames (e.g., averaging or voting)
    aggregated_prediction = np.mean(predictions, axis=0)  # You can use other aggregation techniques based on your requirements
    predicted_label = np.argmax(aggregated_prediction)

    return categories[predicted_label]

def predict_image(image):
    image = cv2.resize(image, (width, height))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = loaded_model.predict(image)
    predicted_label = np.argmax(prediction)
    return categories[predicted_label]

def main():
    st.title('Animal Classification')

    upload_option = st.radio("Choose upload option:", ("Image", "Video"))

    if upload_option == "Image":
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, -1)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Predict'):
                prediction = predict_image(image)
                st.write("Predicted:", prediction)

    elif upload_option == "Video":
        uploaded_video = st.file_uploader("Upload a video...", type=["mp4"])

        if uploaded_video is not None:
            video_path = 'temp_video.mp4'
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            st.video(uploaded_video)

            if st.button('Predict'):
                prediction = predict_video(video_path)
                st.write("Predicted:", prediction)

if __name__ == "__main__":
    main()
