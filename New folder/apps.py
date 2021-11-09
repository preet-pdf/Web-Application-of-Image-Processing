import streamlit as st 
from PIL import Image

import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image as ii
from werkzeug.utils import secure_filename
model =tf.keras.models.load_model('pcmodel.h5',compile=False)
def model_predict(img_path, model):
    img = ii.load_img(img_path, grayscale=False, target_size=(64, 64))
    show_img = ii.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = ii.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    print(preds)
    return preds
def predict(img):
        

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        
        
        img.save('uploads/file.jpg')
        file_path = os.path.join(
            basepath, 'uploads', 'file.jpg')
        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        # x = x.reshape([64, 64]);
        disease_class = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                         'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
                         'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
                         'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
                         'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']
        a = preds[0]
        ind=np.argmax(a)
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        return result



st.title("Super Resolution GAN ")
st.subheader("Upload an image which you want to upscale")   
st.spinner("Testing spinner")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.')
    st.write("")
    if st.button('Upscale Now'):
        st.write("upscaling...") 
        pred = predict(image)
        print(pred)
        st.image(image, caption=pred, use_column_width=True)        
