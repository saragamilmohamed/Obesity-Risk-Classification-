# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from processing1 import preprocess_input  # no backslash, match your module name

encoder_target=joblib.load('target_encoder.pkl')
model=joblib.load('best_xgb_model.pkl')

# === Setup ===
# === Input Form ==
with st.form("user_form"):
    st.subheader("👤 Personal Information")
    col1, col2 = st.columns(2)
    with col1:
        Gender = st.selectbox("👫 Gender", ['Male', 'Female'])
        Age = st.slider("🎂 Age", 10, 100, 25)
        Height = st.number_input("📏 Height (m)", 1.0, 2.5, 1.65)
        Weight = st.number_input("⚖️ Weight (kg)", 30.0, 200.0, 70.0)      
    with col2:
        family_history_with_overweight = st.selectbox(
            "👨‍👩‍👧 Family History with Overweight", ['yes', 'no']
        )
        FAVC = st.selectbox("🍔 Frequent High-Calorie Food", ['yes', 'no'])
        FCVC = st.slider("🥦 Vegetable Consumption Frequency (1-3)", 1.0, 3.0, 2.0)
        NCP = st.slider("🍽️ Meals per Day", 1.0, 5.0, 3.0)

    st.subheader("🏃 Lifestyle & Habits")
    col3, col4 = st.columns(2)
    with col3:
        CAEC = st.selectbox(
            "🍫 Snacking Frequency", ['no', 'Sometimes', 'Frequently', 'Always']
        )
        SMOKE = st.selectbox("🚬 Do you smoke?", ['yes', 'no'])
        CH2O = st.slider("💧 Water Intake (liters)", 0.0, 5.0, 2.0)
    with col4:
        SCC = st.selectbox("📉 Calorie Monitoring", ['yes', 'no'])
        FAF = st.slider("🏋️‍♂️ Physical Activity (0-3)", 0.0, 3.0, 1.0)
        TUE = st.slider("📱 Time on Devices (0-2)", 0.0, 2.0, 1.0)

    CALC = st.selectbox(
        "🍷 Alcohol Consumption", ['no', 'Sometimes', 'Frequently', 'Always']
    )
    MTRANS = st.selectbox(
        "🚌 Main Transport Mode",
        ['Walking', 'Bike', 'Motorbike', 'Public_Transportation', 'Automobile']
    )

    submit = st.form_submit_button("🔍 Predict My Risk!")

    # 📦 Prepare input DataFrame
    user_input = {
        'Gender': [Gender],
        'Age': [Age],
        'Height': [Height],
        'Weight': [Weight],
        'family_history_with_overweight': [family_history_with_overweight],
        'FAVC': [FAVC],
        'FCVC': [FCVC],
        'NCP': [(NCP)],
        'CAEC': [CAEC],
        'SMOKE': [SMOKE],
        'CH2O': [CH2O],
        'SCC': [SCC],
        'FAF': [FAF],
        'TUE': [TUE],
        'CALC': [CALC],
        'MTRANS': [MTRANS]
    }
    input_df = pd.DataFrame(user_input)

    # Add 'id' column first
   # input_df['id'] = 0  # Default value or provide custom value
   #streamlit input_df = input_df[['id'] + [col for col in input_df.columns if col != 'id']]  # Reorder to place 'id' first

# === Prediction Section ===
processed = preprocess_input(input_df)
print(processed)
if submit:
    try:
        prediction = model.predict(processed)
        predicted_label = encoder_target.inverse_transform(prediction)[0]

        st.success(f"🎉 Your Predicted Obesity Risk Level is: **{predicted_label}**")
        st.balloons()
        st.markdown("💚 Stay active, eat healthy, and take care of yourself!")
    except Exception as e:
        st.error(f"⚠️ Something went wrong during prediction:\n{e}")
