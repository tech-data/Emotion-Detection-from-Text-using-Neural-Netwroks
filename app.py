import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
from preprocess import *

st.write("""
# Emotion detection app

This app detects the emotion of a given text

Data obtained from Kaggle: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp.
""")

st.sidebar.header('User Input')




def user_input():
    text = st.sidebar.text_input('Enter your sentence: ')

    return text
input = user_input()

st.write(input)
encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))


model=tf.keras.models.load_model('my_model.h5')
input=preprocess(input)

array = cv.transform([input]).toarray()

pred = model.predict(array)
a=np.argmax(pred, axis=1)
prediction = encoder.inverse_transform(a)[0]


st.subheader('Prediction')
if input == '':
    st.write('The emotion of this text is...')
else:

    st.write(prediction)

