import streamlit as st 
from PIL import Image
import numpy as np
from ISR.models import RRDN
from flask import Flask, render_template, request
import cv2
from keras.models import load_model


import numpy as np
from tensorflow.keras.applications import ResNet50 

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from tensorflow.keras.models import Sequential, Model
from keras.utils import np_utils
from tensorflow.keras.preprocessing import image, sequence
import cv2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

vocab = np.load('vocab1.npy', allow_pickle=True)

vocab = vocab.item()

inv_vocab = {v:k for k,v in vocab.items()}


print("+"*50)
print("vocabulary loaded")


embedding_size = 128
vocab_size = len(vocab)
max_len = 40


image_model = Sequential()

image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))
image_model.add(RepeatVector(max_len))


language_model = Sequential()

language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(embedding_size)))


conca = Concatenate()([image_model.output, language_model.output])
x = LSTM(128, return_sequences=True)(conca)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocab_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[image_model.input, language_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.load_weights('mine_model_weights1.h5')

print("="*150)
print("MODEL LOADED")


resnet = load_model('C:/Users/LENOVO/Downloads/model_name.h5')
print("="*150)
print("RESNET MODEL LOADED")




def predict(img):
    
    global model, resnet, vocab, inv_vocab

    

    img.save('static/file.jpg')

    print("="*50)
    print("IMAGE SAVED")


    
    image = cv2.imread('static/file.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224,224))

    image = np.reshape(image, (1,224,224,3))

    
    
    incept = resnet.predict(image).reshape(1,2048)

    print("="*50)
    print("Predict Features")


    text_in = ['startofseq']

    final = ''

    print("="*50)
    print("GETING Captions")

    count = 0
    while tqdm(count < 20):

        count += 1

        encoded = []
        for i in text_in:
            encoded.append(vocab[i])

        padded = pad_sequences([encoded], maxlen=max_len, padding='post', truncating='post').reshape(1,max_len)

        sampled_index = np.argmax(model.predict([incept, padded]))

        sampled_word = inv_vocab[sampled_index]

        if sampled_word != 'endofseq':
            final = final + ' ' + sampled_word

        text_in.append(sampled_word)



    return final

import streamlit as st
import numpy as np
import pandas as pd


def app():
    

    st.title("Image Captioning Using Machine Learning")
    st.subheader("Upload an image which you want to Predict")   
    st.spinner("Testing spinner")

    uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.')
        st.write("")
        if st.button('Predict Now'):
            st.write("Predicting...") 
            pred = predict(image)
            print(pred)
            st.image(image, caption=pred, use_column_width=True)        
