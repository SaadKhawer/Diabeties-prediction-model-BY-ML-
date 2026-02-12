import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("diabetes_model.pkl", "rb"))

st.set_page_config(page_title="Diabetes Prediction AI", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient details to predict diabetes")

preg = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 300, 120)
bp = st.number_input("Blood Pressure", 0, 200, 80)
skin = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")
