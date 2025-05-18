# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from processing import preprocess_input  # no backslash, match your module name

# === Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_pickle(filename):
    """Helper function to load pickled files safely."""
    filepath = os.path.join(BASE_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return joblib.load(f)
    else:
        st.error(f"❌ File not found: {filename}")
        raise FileNotFoundError(f"{filename} not found in {BASE_DIR}")

# === Load Model & Target Encoder ===
model = load_pickle('best_xgb_model.pkl')
encoder_target = load_pickle('target_encoder.pkl')

# === Sidebar ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3875/3875029.png", width=100)
    st.markdown("## 🤖 About the App")
    st.markdown("""
        This app predicts your **Obesity Risk Level** based on your lifestyle and habits using a trained **Machine Learning** model.
        Fill out the form and click **Predict** to discover your risk level!
    """)

# === Main Title ===
st.markdown(
    "<h1 style='text-align: center; color: #FF4B4B;'>🍕 Obesity Risk Predictor 🍎</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center; color: gray;'>A smarter way to understand your health.</h5>",
    unsafe_allow_html=True
)
st.write("---")

# === Input Form ===
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
        "🍷 Alcohol Consumption", ['no', 'Sometimes', 'Frequently']
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
        'NCP': [NCP],
        'CAEC': [CAEC],
        'SMOKE': [SMOKE],
        'CH2O': [CH2O],
        'SCC': [SCC],
        'FAF': [FAF],
        'TUE': [TUE],
        'CALC': [CALC],
        'MTRANS': [MTRANS],
    }
    # input_df = pd.DataFrame(preprocess_input(user_input))
    
processed = preprocess_input(user_input)
# === Prediction Section ===
if submit:
    try:
        prediction = model.predict(processed)
        predicted_label = encoder_target.inverse_transform(prediction)[0]

                # Map prediction labels to background colors
        background_colors = {
        'normal_weight': '28a745',           # dark green
        'insufficient_weight': '17a2b8',     # teal blue
        'overweight_level_i': 'ffc107',      # strong amber
        'overweight_level_ii': 'ffb300',     # deeper amber
        'obesity_type_i': 'dc3545',          # dark red
        'obesity_type_ii': 'bd2130',         # deeper red
        'obesity_type_iii': 'a71d2a'         # very dark red
        }

        bg_color = background_colors.get(predicted_label.lower(), '6c757d')  # fallback: gray

        st.markdown(
            f"""
            <div style='background-color: #{bg_color}; padding: 20px; border-radius: 10px;'>
                <h4 style='color: white; font-size: 20px;'>Your Predicted Obesity Risk Level is: 
                <b>{predicted_label.replace('_', ' ').title()}</b></h4>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.balloons()
        st.markdown("💚 Stay active, eat healthy, and take care of yourself!")
    except Exception as e:
        st.error(f"⚠️ Something went wrong during prediction:\n{e}")
