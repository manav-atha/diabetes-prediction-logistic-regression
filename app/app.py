import os
import pickle

import numpy as np
import streamlit as st

base_path = os.path.dirname(__file__)

model_path = os.path.join(base_path, "model.pkl")
scaler_path = os.path.join(base_path, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

st.title("Diabetes Prediction App")

pregnancies = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("Diabetes Pedigree Function")
age = st.number_input("Age")

input_data = np.array([[pregnancies, glucose, bp, skin,
                        insulin, bmi, dpf, age]])

if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("High chance of Diabetes")
    else:
        st.success("Low chance of Diabetes")
        