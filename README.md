# Customer Churn Prediction

This project is a machine learning application that predicts whether a bank customer is likely to churn. It uses a TensorFlow artificial neural network trained on customer account and demographic data, with a Streamlit interface for interactive predictions.

## Project Summary

The model uses the following customer information:

- Credit score
- Geography and gender
- Age and tenure
- Account balance
- Number of products
- Credit card ownership
- Active membership status
- Estimated salary

Categorical values are encoded with `LabelEncoder` and `OneHotEncoder`. Numeric features are standardized with `StandardScaler` before being passed to the trained neural network. The app returns a churn probability and a simple interpretation of the prediction.

## Features

- Interactive Streamlit form for customer details
- Reusable trained TensorFlow model
- Saved encoders and scaler for consistent inference
- Training script with validation, early stopping, and TensorBoard logging

## Run Locally

Requires Python 3.10 to 3.13. Using `uv` is recommended:

```bash
uv sync
uv run streamlit run app.py
```

Then open `http://localhost:8501` in a browser.

With an activated virtual environment, the equivalent command is:

```bash
streamlit run app.py
```

## Training

To retrain the model and regenerate the preprocessing artifacts:

```bash
uv run python experiment.py
```

The training script reads `Data.csv`, saves the model as `customer_churn_model.h5`, and writes the encoders and scaler as `.pkl` files. TensorBoard logs are written under `logs/`.

## Repository Structure

```text
app.py                    Streamlit prediction interface
experiment.py             Data preprocessing and model training
trained_model.py          Example prediction script
Data.csv                  Training dataset
customer_churn_model.h5   Trained TensorFlow model
*_encoder.pkl             Saved categorical encoders
scaler.pkl                Saved feature scaler
pyproject.toml            Project metadata and dependencies
uv.lock                   Locked dependency versions
```


