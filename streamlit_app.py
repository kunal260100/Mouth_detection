import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import cv2

# Load the pre-trained model
model_path = 'model.keras'
model = load_model(model_path)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function for mouth detection
def detect_mouth(image):
    try:
        resized_image = tf.image.resize(image, (256, 256))
        resized_image_np = resized_image.numpy()
        resized_image_np = np.expand_dims(resized_image_np / 255, 0)
        yhat = model.predict(resized_image_np)

        st.image(resized_image_np[0], caption='Uploaded Image', width=300)
        st.write('Prediction:', 'Open' if yhat > 0.5 else 'Close')

    except Exception as e:
        st.error(f'Error: {e}')

# Streamlit app
def main():
    st.title('Mouth Detector')

    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=300)
        st.write('')
        st.write('Classifying...')

        detect_mouth(image)

if __name__ == '__main__':
    main()
