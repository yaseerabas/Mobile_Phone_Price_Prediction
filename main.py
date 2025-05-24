import pandas as pd
import streamlit as st
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('model.h5')

with open('Scaler_X.pkl', 'rb') as file:
    Scale_X = pickle.load(file)

with open('Scaler_y.pkl', 'rb') as file:
    Scale_y = pickle.load(file)

st.title("Predict Mobile phone Price")

resolution = st.number_input("Enter Resolution")

ppi = st.number_input("Pixel Per Inch")

core = st.slider("CPU Core", 0, 8)

freq = st.number_input("CPU Frequency")

memory = float(st.selectbox("Internel Memory", (4, 8, 16, 32, 64, 128)))

ram = st.number_input("RAM")

rearCam = float(st.slider("Rear Camera", 0, 23))

frontCam = float(st.slider("Front Camera", 0, 20))

battery = st.number_input("Battery (mAH)", min_value=800, max_value=9500, value=800)

thickness = st.number_input("Tickness", min_value=5.1, max_value=18.5, value=5.1)


input_data = pd.DataFrame({
    'resoloution': [resolution],
    'ppi': [ppi],
    'cpu core': [core],
    'cpu freq': [freq],
    'internal mem': [memory],
    'ram': [ram],
    'RearCam': [rearCam],
    'Front_Cam': [frontCam],
    'battery': [battery],
    'thickness': [thickness]
})

prediction = model.predict(input_data)

if st.button("Predict"):
    st.success(f"Prediction: {prediction[0][0]:.2f}")
