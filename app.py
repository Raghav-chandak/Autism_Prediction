import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

# Page title
st.title("Autism Spectrum Disorder (ASD) Prediction App")
st.markdown("This app predicts whether an individual is likely to have Autism based on behavioral and demographic inputs.")

# Create form
with st.form("autism_form"):
    st.subheader("Behavioral Assessment Scores (A1 to A10)")
    A1_Score = st.selectbox("A1 Score", [0, 1])
    A2_Score = st.selectbox("A2 Score", [0, 1])
    A3_Score = st.selectbox("A3 Score", [0, 1])
    A4_Score = st.selectbox("A4 Score", [0, 1])
    A5_Score = st.selectbox("A5 Score", [0, 1])
    A6_Score = st.selectbox("A6 Score", [0, 1])
    A7_Score = st.selectbox("A7 Score", [0, 1])
    A8_Score = st.selectbox("A8 Score", [0, 1])
    A9_Score = st.selectbox("A9 Score", [0, 1])
    A10_Score = st.selectbox("A10 Score", [0, 1])

    st.subheader("Personal Details")
    age = st.slider("Age", 1, 100, 25)
    gender = st.selectbox("Gender", encoders['gender'].classes_.tolist())
    ethnicity = st.selectbox("Ethnicity", encoders['ethnicity'].classes_.tolist())
    jaundice = st.selectbox("Had Jaundice as a baby?", encoders['jaundice'].classes_.tolist())
    austim = st.selectbox("Has previous autism diagnosis?", encoders['austim'].classes_.tolist())
    contry_of_res = st.selectbox("Country of Residence", encoders['contry_of_res'].classes_.tolist())
    used_app_before = st.selectbox("Used screening app before?", encoders['used_app_before'].classes_.tolist())
    result = st.selectbox("Screening Score Result (0 or 1)", [0, 1])
    relation = st.selectbox("Who is filling this form?", encoders['relation'].classes_.tolist())

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame from inputs
    input_data = pd.DataFrame([[ 
        A1_Score, A2_Score, A3_Score, A4_Score, A5_Score,
        A6_Score, A7_Score, A8_Score, A9_Score, A10_Score,
        age, gender, ethnicity, jaundice, austim,
        contry_of_res, used_app_before, result, relation
    ]], columns=[
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'ethnicity', 'jaundice', 'austim',
        'contry_of_res', 'used_app_before', 'result', 'relation'
    ])

    # Encode categorical fields using stored encoders
    for col in ['gender', 'ethnicity', 'jaundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']:
        input_data[col] = encoders[col].transform(input_data[col])

    # Predict
    prediction = model.predict(input_data)[0]

    # Show result
    if prediction == 1:
        st.error("🔴 The model predicts a **HIGH risk of Autism Spectrum Disorder (ASD)**.")
    else:
        st.success("🟢 The model predicts a **LOW risk of Autism Spectrum Disorder (ASD)**.")
