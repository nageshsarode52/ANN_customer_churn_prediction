import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
print("Importing libraries completed!")

#data loading
data = pd.read_csv('data.csv')
print("Data loaded successfully!")


# Preprocessing
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

#Ecoding categorical variables
label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
print("Label encoding for Gender completed!")


#OneHot encoding for Geography
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False)
geography_encoded = onehot_encoder.fit_transform(data[['Geography']])
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder.get_feature_names_out(['Geography']))
data = pd.concat([data, geography_df], axis=1)
data = data.drop('Geography', axis=1)
print("One-hot encoding for Geography completed!")

# save the encoders and scaler
with open('label_encoder_gender.pkl', 'wb') as f:
    pickle.dump(label_encoder_gender, f)
with open('onehot_encoder.pkl', 'wb') as f:
    pickle.dump(onehot_encoder, f)
print("Encoders saved successfully!")

# Divide the data into independent and dependent variables
X = data.drop('Exited', axis=1)
y = data['Exited']
print("Data divided into independent and dependent variables completed!")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets completed!")

#Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed!")

# save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved successfully!")

# Create ANN model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
print("Importing TensorFlow libraries completed!")

# Build the model - ANN

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
print("ANN model created successfully!")
print(model.summary())


# compile the model
import tensorflow  
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
print("Model compilation completed!")


# setup TensorBoard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
print("TensorBoard setup completed!")

# setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("Early stopping setup completed!")

# Training model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, 
                    callbacks=[EarlyStopping(monitor='val_loss', patience=10), 
                    TensorBoard(log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))])
print("Model training completed!")

model.save('customer_churn_model.h5')
print("Model saved successfully!")


