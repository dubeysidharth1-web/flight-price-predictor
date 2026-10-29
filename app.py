import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="centered")

st.title("✈️ Flight Price Prediction App")
st.write("Apni flight details enter kijiye aur AI-estimated price dekhiye.")

# 1. Model aur feature columns load karein
@st.cache_resource
def load_model():
    model = joblib.load('flight_price_xgb_model.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, cols

model, model_columns = load_model()

# 2. User Inputs (Form Design)
col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox("Airline Chuniye", ['Vistara', 'Air India', 'IndiGo', 'SpiceJet', 'AirAsia'])
    source = st.selectbox("From (Kahan se)", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    stops = st.selectbox("Stops", [0, 1, 2])

with col2:
    destination = st.selectbox("To (Kahan tak)", ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'])
    duration_hours = st.slider("Flight Duration (Hours)", min_value=1.0, max_value=25.0, value=2.5, step=0.5)
    days_left = st.slider("Kitne din bache hain booking mein?", min_value=1, max_value=50, value=15)

# Source aur destination same nahi hone chahiye
if source == destination:
    st.warning("⚠️ Source aur Destination same nahi ho sakte!")

# 3. Predict Button
if st.button("Estimate Price 🎫"):
    if source != destination:
        # A. Engineered features
        is_last_minute = 1 if days_left <= 5 else 0
        duration_per_stop = duration_hours / (stops + 1)
        
        # B. Empty row banayein
        input_data = pd.DataFrame(0, index=[0], columns=model_columns)
        
        # C. Numerical values fill karein
        input_data['stops'] = stops
        input_data['duration_hours'] = duration_hours
        input_data['days_left'] = days_left
        input_data['is_last_minute'] = is_last_minute
        input_data['duration_per_stop'] = duration_per_stop
        
        # D. Categorical values set karein
        airline_col = f"airline_{airline}"
        source_col = f"source_{source}"
        dest_col = f"destination_{destination}"
        
        if airline_col in input_data.columns:
            input_data[airline_col] = 1
        if source_col in input_data.columns:
            input_data[source_col] = 1
        if dest_col in input_data.columns:
            input_data[dest_col] = 1
            
        # E. Prediction
        predicted_price = model.predict(input_data)[0]
        
        st.success(f"Estimated Ticket Price: **₹{predicted_price:,.0f}**")
        
        if is_last_minute:
            st.info("💡 Yeh flight last-minute category mein aati hai, isliye daam zyada hain.")
            
