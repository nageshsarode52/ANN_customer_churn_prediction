import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model


# Load the trained model
model = tf.keras.models.load_model('customer_churn_model.h5')

# Load the scaler, encoders and OneHotEncoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields for user to enter customer data
geography = st.selectbox("Geography", onehot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_) 
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance", min_value=0.0, step=0.01)
credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, step=0.01)
tenure = st.slider("Tenure (years)", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1]) 

# Prepare the input data for prediction
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encoding for Geography
geography_encoded = onehot_encoder.transform([[geography]])
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Prediction Churn
prediction = model.predict(input_data_scaled)
st.write("Prediction Probability: ", prediction[0][0])
if prediction[0][0] > 0.5:
    st.success("---> The customer is likely to churn.")
else:
    st.success("---> The customer is not likely to churn.")

