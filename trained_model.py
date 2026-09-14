# import libries
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
print("Importing libraries completed!")

# Load the trained model
model = load_model('customer_churn_model.h5')
print("Model loaded successfully!")

# Load the scaler and encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)
print("Scaler and encoders loaded successfully!")


input_data = {
    'CreditScore': 200,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 0,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 0
}

# One-hot encoding for Geography
geography_encoded = onehot_encoder.transform([[input_data['Geography']]])
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))

# Combine one-hot encoded geography with the input data
input_data = pd.DataFrame([input_data])
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)
 # drop the original Geography column
input_data = input_data.drop('Geography', axis=1)
print("One-hot encoding for Geography completed!")

# label encoding for Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
print("Label encoding for Gender completed!")

# Scale the input data
input_data = scaler.transform(input_data)
print("Feature scaling completed!") 

# Make prediction
prediction = model.predict(input_data)
if prediction[0][0] > 0.5:
    print("-------> The customer is likely to churn.")
else:
    print("-------> The customer is not likely to churn.")   


