import streamlit as st
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the pre-trained encoders, scaler, and ANN model
with open('pickle_files/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('pickle_files/onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('pickle_files/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

ann_model = load_model('model.h5')

# Streamlit app
def main():
    st.title("Customer Churn Prediction")

    st.header("Input Customer Details")

    # Input fields
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
    balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

    # Process input data
    if st.button("Predict Churn"):
        # Convert inputs to a dictionary
        input_data = {
            'CreditScore': credit_score,
            'Gender': label_encoder_gender.transform([gender])[0],
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': 1 if has_cr_card == "Yes" else 0,
            'IsActiveMember': 1 if is_active_member == "Yes" else 0,
            'EstimatedSalary': estimated_salary
        }

        # One-hot encode geography
        geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

        # Combine input data and one-hot encoded features
        input_data_df = pd.DataFrame([input_data])
        input_data_df = pd.concat([input_data_df, geo_encoded_df], axis=1)

        # Ensure columns align with the training set
        input_data_df = input_data_df.drop(columns=['Geography'], errors='ignore')

        # Scale features
        scaled_data = scaler.transform(input_data_df)

        # Make prediction
        prediction = ann_model.predict(scaled_data)[0][0]
        prediction_label = "Churn" if prediction > 0.5 else "No Churn"

        # Display result
        if prediction > 0.5:
            st.header("Prediction: Customer is likely to churn.")
        else:
            st.header("Prediction: Customer is not likely to churn.")

if __name__ == "__main__":
    main()
